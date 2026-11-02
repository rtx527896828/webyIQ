"""GPTQ for the webyIQ int4 format (asymmetric, group 32, f16 scale/min).

The quantization grid is exactly the one of convert.py, so the engine runs the result unchanged;
GPTQ only chooses better integer codes and group ranges by compensating each column's rounding
error on the not-yet-quantized columns (Frantar et al., 2022).

Layers are processed in order: calibration activations are captured with the original layer,
the layer's linears are quantized, and the quantized layer produces the next layer's inputs.

  uv run python gptq.py <hf_model_dir> <out_dir> [--nsamples 128 --seqlen 512]
  uv run python convert.py <hf_model_dir> <dst> --prequant <out_dir> --search

Writes <out_dir>/<tensor>.npz with q (uint8 [N, K]), scale and mn (f16 [N, K/32]).
"""

import argparse
import io
import os
import time
import urllib.request

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from convert import quantize_q4

WIKITEXT_TRAIN = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/train-00000-of-00001.parquet"
GROUP = 32


def calibration(model_dir, nsamples, seqlen, seed=0):
    import pyarrow.parquet as pq

    raw = urllib.request.urlopen(WIKITEXT_TRAIN).read()
    text = "".join(pq.read_table(io.BytesIO(raw)).column("text").to_pylist())
    ids = AutoTokenizer.from_pretrained(model_dir)(text, add_special_tokens=False)["input_ids"]
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(ids) - seqlen, nsamples)
    return torch.tensor([ids[s : s + seqlen] for s in starts])


def gptq(W, H, block=128, damp=0.01):
    """W [N, K] f32, H [K, K] f64 (mean of x x^T). Returns q uint8 [N, K], scale/mn f16 [N, K/32]."""
    W = W.clone().float()
    N, K = W.shape
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(K, dtype=H.dtype)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True).float()

    Q = torch.zeros(N, K, dtype=torch.uint8)
    scale = torch.zeros(N, K // GROUP, dtype=torch.float16)
    mn = torch.zeros(N, K // GROUP, dtype=torch.float16)
    for i1 in range(0, K, block):
        i2 = min(i1 + block, K)
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % GROUP == 0:
                # grid from the current (error-updated) values of this group, as convert.q4_grid
                g = W1[:, i : i + GROUP]
                lo, hi = g.min(1).values, g.max(1).values
                s16, m16 = ((hi - lo) / 15.0).half(), lo.half()
                scale[:, col // GROUP], mn[:, col // GROUP] = s16, m16
                sf, mf = s16.float(), m16.float()
                safe = torch.where(sf == 0, torch.ones_like(sf), sf)
            w = W1[:, i]
            q = torch.clamp(torch.round((w - mf) / safe), 0, 15)
            Q[:, col] = q.to(torch.uint8)
            err = (w - (q * sf + mf)) / Hinv1[i, i]
            W1[:, i:] -= err[:, None] * Hinv1[i, i:][None, :]
            Err1[:, i] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return Q, scale, mn


def dequant(q, scale, mn):
    s = scale.float().repeat_interleave(GROUP, 1)
    m = mn.float().repeat_interleave(GROUP, 1)
    return q.float() * s + m


class HessianHook:
    def __init__(self, module):
        k = module.in_features
        self.H = torch.zeros(k, k, dtype=torch.float64)
        self.n = 0
        self.handle = module.register_forward_pre_hook(self)

    def __call__(self, _mod, inputs):
        x = inputs[0].reshape(-1, inputs[0].shape[-1]).double()
        self.H += x.T @ x
        self.n += x.shape[0]

    def done(self):
        self.handle.remove()
        return self.H / self.n


def run_layer(layer, hidden, pos_emb, bs):
    out = []
    for b in range(0, hidden.shape[0], bs):
        h = layer(hidden[b : b + bs], attention_mask=None, position_embeddings=pos_emb)
        out.append(h[0] if isinstance(h, tuple) else h)
    return torch.cat(out)


def quantize_linears(out_dir, name, mods, H):
    """GPTQ the row-concatenation of `mods` (same input), save, write dequantized weights back."""
    W = torch.cat([m.weight.data for m in mods])
    t = time.time()
    q, scale, mn = gptq(W, H)
    np.savez(os.path.join(out_dir, name + ".npz"), q=q.numpy(), scale=scale.numpy(), mn=mn.numpy())
    deq = dequant(q, scale, mn)
    err = ((deq - W) @ H.float() * (deq - W)).sum().item() / W.shape[0]
    r = 0
    for m in mods:
        n = m.weight.shape[0]
        m.weight.data.copy_(deq[r : r + n])
        r += n
    print(f"  {name:<14} {tuple(W.shape)} {time.time() - t:5.1f}s  weighted err/row {err:.3e}", flush=True)


def text_model(model):
    """The decoder stack (Qwen3.5 nests it under `language_model`)."""
    m = model.model
    return getattr(m, "language_model", m)


def positions(tm, hidden):
    seq = hidden.shape[1]
    if "qwen3_5" in type(tm).__name__.lower():
        # multimodal rope with identical text positions on all three sections
        pos = torch.arange(seq)[None, None].expand(3, 1, seq)
    else:
        pos = torch.arange(seq)[None]
    return tm.rotary_emb(hidden[:1], pos)


def linear_groups(li, layer):
    """(engine tensor name, linears sharing one input) of a decoder layer, in engine order."""
    m = layer.mlp
    if hasattr(layer, "linear_attn"):
        a = layer.linear_attn
        mixer = [(f"l{li}.in_proj", [a.in_proj_qkv, a.in_proj_z, a.in_proj_b, a.in_proj_a]), (f"l{li}.o", [a.out_proj])]
    else:
        a = layer.self_attn
        mixer = [(f"l{li}.qkv", [a.q_proj, a.k_proj, a.v_proj]), (f"l{li}.o", [a.o_proj])]
    return mixer + [(f"l{li}.gate_up", [m.gate_proj, m.up_proj]), (f"l{li}.down", [m.down_proj])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--check", action="store_true", help="only verify the layer-by-layer forward")
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    os.makedirs(args.out, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    tm = text_model(model)
    tied = model.lm_head.weight is tm.embed_tokens.weight
    ids = calibration(args.model, args.nsamples, args.seqlen)
    print(f"calibration: {tuple(ids.shape)} tokens from wikitext-2 train; tied embeddings: {tied}", flush=True)

    if args.check:
        with torch.no_grad():
            x = ids[:2, :64]
            h = tm.embed_tokens(x)
            pe = positions(tm, h)
            for layer in tm.layers:
                h = run_layer(layer, h, pe, 2)
            mine = model.lm_head(tm.norm(h))
            ref = model(x).logits
            print("layer-by-layer vs model forward: max |diff|", (mine - ref).abs().max().item())
        return

    with torch.no_grad():
        # the engine's embedding is RTN with clip search (not a matmul, so no GPTQ)
        emb = tm.embed_tokens.weight
        emb.data.copy_(torch.from_numpy(quantize_q4(emb.data.numpy(), GROUP, search=True)[2]))
        hidden = tm.embed_tokens(ids)
        pos_emb = positions(tm, hidden)

        t0 = time.time()
        for li, layer in enumerate(tm.layers):
            groups = linear_groups(li, layer)
            hooks = [HessianHook(mods[0]) for _, mods in groups]
            run_layer(layer, hidden, pos_emb, args.batch)
            print(f"layer {li} ({time.time() - t0:.0f}s)", flush=True)
            for (name, mods), hook in zip(groups, hooks):
                quantize_linears(args.out, name, mods, hook.done())
            hidden = run_layer(layer, hidden, pos_emb, args.batch)

        if tied:
            # lm_head shares the embedding, which keeps its RTN codes (GPTQ would tune them for
            # lm_head inputs at the expense of the embedding lookup)
            print("tied lm_head: kept RTN")
        else:
            # lm_head input statistics directly (running lm_head would materialize [tokens, vocab] logits)
            H = torch.zeros(hidden.shape[-1], hidden.shape[-1], dtype=torch.float64)
            for b in range(0, hidden.shape[0], args.batch):
                x = tm.norm(hidden[b : b + args.batch]).reshape(-1, hidden.shape[-1]).double()
                H += x.T @ x
            quantize_linears(args.out, "lm_head", [model.lm_head], H / (hidden.shape[0] * hidden.shape[1]))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

"""Convert a HF checkpoint (Llama architecture, or the Qwen3.5 text model) into the webyIQ int4 format.

Layout produced (all little endian):
  manifest.json        model config + tensor table
  shard-XXX.bin        raw tensor bytes, each tensor 256-byte aligned

Quantized linear weight [N, K] (row = output channel) is stored as two tensors:
  <name>.q  u32[N, K/8]          8 nibbles per u32; u32 i holds weights 8i..8i+7 with
                                 byte b = w[8i+b] | w[8i+4+b] << 4, so (u & 0x0F0F0F0F)
                                 and (u >> 4 & 0x0F0F0F0F) are 4 consecutive weights each
                                 (ready for dot4I8Packed against int8 activations)
  <name>.s  u32[N, K/group]      pack2x16float(scale, min); w = q * scale + min
"""

import argparse
import hashlib
import json
import os
import struct

import numpy as np

SHARD_BYTES = 128 << 20
ALIGN = 256


def load_safetensors(path):
    """Return {name: (dtype, shape, memmap-slice)} without copying data."""
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    base = 8 + hlen
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        s, e = meta["data_offsets"]
        out[name] = (meta["dtype"], meta["shape"], mm[base + s : base + e])
    return out


def to_f32(entry):
    dtype, shape, raw = entry
    if dtype == "BF16":
        u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return u.view(np.float32).reshape(shape)
    if dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    raise ValueError(dtype)


def q4_grid(lo, hi):
    """Asymmetric 4-bit grid for a group range: (scale, min) rounded to f16 as stored."""
    return ((hi - lo) / 15.0).astype(np.float16), lo.astype(np.float16)


def q4_round(g, scale, mn):
    """Quantize values g [..., group] on the grid -> (q uint, dequant f32)."""
    scale_f = scale.astype(np.float32)[..., None]
    mn_f = mn.astype(np.float32)[..., None]
    q = np.clip(np.rint((g - mn_f) / np.where(scale_f == 0, 1.0, scale_f)), 0, 15)
    return q.astype(np.uint32), q.astype(np.float32) * scale_f + mn_f


def pack_q4(q, scale, mn):
    """q [N, K] ints in 0..15, scale/min f16 [N, K/g] -> (packed u32 [N, K/8], scales u32 [N, K/g])."""
    n, k = q.shape
    q = q.astype(np.uint32).reshape(n, k // 8, 8)
    shifts = np.array([0, 8, 16, 24, 4, 12, 20, 28], dtype=np.uint32)[None, None, :]
    packed = np.bitwise_or.reduce(q << shifts, axis=2).astype(np.uint32)
    sc = scale.view(np.uint16).astype(np.uint32) | (mn.view(np.uint16).astype(np.uint32) << 16)
    return packed, sc.astype(np.uint32)


def quantize_q4(w, group, search=False):
    """Asymmetric round-to-nearest int4. With `search`, each group's range is shrunk by the
    factor (1.0 .. 0.7) that minimizes its squared error. Returns (packed, scales, dequant f32)."""
    n, k = w.shape
    assert k % group == 0 and group % 8 == 0
    q_all = np.empty((n, k), np.uint32)
    scale = np.empty((n, k // group), np.float16)
    mn = np.empty((n, k // group), np.float16)
    deq = np.empty((n, k), np.float32)
    rows = max(1, (1 << 22) // k)  # bound temporary memory
    for r0 in range(0, n, rows):
        g = w[r0 : r0 + rows].reshape(-1, k // group, group)
        lo, hi = g.min(axis=2), g.max(axis=2)
        best_s, best_m = q4_grid(lo, hi)
        if search:
            best_err = np.full(lo.shape, np.inf, np.float32)
            for f in np.linspace(1.0, 0.7, 16):
                s_, m_ = q4_grid(lo * f, hi * f)
                err = ((q4_round(g, s_, m_)[1] - g) ** 2).sum(axis=2)
                better = err < best_err
                best_err = np.where(better, err, best_err)
                best_s = np.where(better, s_, best_s)
                best_m = np.where(better, m_, best_m)
        q, d = q4_round(g, best_s, best_m)
        q_all[r0 : r0 + rows] = q.reshape(-1, k)
        deq[r0 : r0 + rows] = d.reshape(-1, k)
        scale[r0 : r0 + rows], mn[r0 : r0 + rows] = best_s, best_m
    packed, sc = pack_q4(q_all, scale, mn)
    return packed, sc, deq


def shard_hashes(out_dir, shards):
    """Short content hashes of the shard files (browser cache keys: a re-converted model never
    reuses stale cached shards)."""
    out = []
    for name in shards:
        h = hashlib.sha256()
        with open(os.path.join(out_dir, name), "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
        out.append(h.hexdigest()[:16])
    return out


class ShardWriter:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.index = -1
        self.f = None
        self.pos = 0
        self.tensors = {}
        self._next()

    def _next(self):
        if self.f:
            self.f.close()
        self.index += 1
        self.f = open(os.path.join(self.out_dir, f"shard-{self.index:03d}.bin"), "wb")
        self.pos = 0

    def add(self, name, arr, dtype, shape):
        data = np.ascontiguousarray(arr).tobytes()
        if self.pos > 0 and self.pos + len(data) > SHARD_BYTES:
            self._next()
        pad = (-self.pos) % ALIGN
        self.f.write(b"\0" * pad)
        self.pos += pad
        self.tensors[name] = {
            "shard": self.index,
            "offset": self.pos,
            "bytes": len(data),
            "dtype": dtype,
            "shape": list(shape),
        }
        self.f.write(data)
        self.pos += len(data)

    def add_q4(self, name, w, group, search=False, prequant=None):
        """prequant: directory of `<name>.npz` (q, scale, mn) from gptq.py, used when present."""
        path = prequant and os.path.join(prequant, name + ".npz")
        if path and os.path.exists(path):
            z = np.load(path)
            assert z["q"].shape == w.shape, (name, z["q"].shape, w.shape)
            packed, sc = pack_q4(z["q"], z["scale"], z["mn"])
        else:
            packed, sc, _ = quantize_q4(w, group, search)
        self.add(name + ".q", packed, "q4", w.shape)
        self.add(name + ".s", sc, "q4s", (w.shape[0], w.shape[1] // group))

    def close(self):
        self.f.close()
        return self.index + 1


def write_llama(w, cfg, t, add_q4):
    hd = cfg["hidden_size"]
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        print(f"layer {i}", flush=True)
        w.add(f"l{i}.attn_norm", t(p + "input_layernorm.weight"), "f32", [hd])
        qkv = np.concatenate(
            [t(p + "self_attn.q_proj.weight"), t(p + "self_attn.k_proj.weight"), t(p + "self_attn.v_proj.weight")]
        )
        add_q4(f"l{i}.qkv", qkv)
        add_q4(f"l{i}.o", t(p + "self_attn.o_proj.weight"))
        w.add(f"l{i}.ffn_norm", t(p + "post_attention_layernorm.weight"), "f32", [hd])
        add_q4(f"l{i}.gate_up", np.concatenate([t(p + "mlp.gate_proj.weight"), t(p + "mlp.up_proj.weight")]))
        add_q4(f"l{i}.down", t(p + "mlp.down_proj.weight"))
    w.add("norm", t("model.norm.weight"), "f32", [hd])
    tied = bool(cfg.get("tie_word_embeddings"))
    if not tied:
        print("lm_head")
        add_q4("lm_head", t("lm_head.weight"))
    eos = cfg["eos_token_id"]
    return {
        "hidden_size": hd,
        "intermediate_size": cfg["intermediate_size"],
        "num_layers": cfg["num_hidden_layers"],
        "num_heads": cfg["num_attention_heads"],
        "num_kv_heads": cfg["num_key_value_heads"],
        "head_dim": cfg.get("head_dim", hd // cfg["num_attention_heads"]),
        "vocab_size": cfg["vocab_size"],
        "rms_norm_eps": cfg["rms_norm_eps"],
        "rope_theta": cfg["rope_theta"],
        "max_position_embeddings": cfg["max_position_embeddings"],
        "bos_token_id": cfg["bos_token_id"],
        "eos_token_ids": eos if isinstance(eos, list) else [eos],
        "tie_word_embeddings": tied,
    }


def write_qwen35(w, cfg, t, add_q4, src):
    """Qwen3.5 text model: Gated DeltaNet ("linear") and gated full-attention layers.
    Norms with a zero-centered weight are stored as (1 + w); the vision tower is skipped.
    The MTP head (draft model for speculative decoding) is stored as `mtp.*`: its decoder layer under
    the same names as a full-attention layer, `fc` over cat(norm_e(embedding), norm_h(hidden))."""
    c = cfg["text_config"]
    hd = c["hidden_size"]
    pre = "model.language_model."

    def one_plus(name):
        return 1.0 + t(name)

    def full_attention(name, p):
        a = p + "self_attn."
        # q_proj rows are per head [q (head_dim) | gate (head_dim)], followed by k and v
        add_q4(f"{name}.qkv", np.concatenate([t(a + "q_proj.weight"), t(a + "k_proj.weight"), t(a + "v_proj.weight")]))
        w.add(f"{name}.q_norm", one_plus(a + "q_norm.weight"), "f32", [c["head_dim"]])
        w.add(f"{name}.k_norm", one_plus(a + "k_norm.weight"), "f32", [c["head_dim"]])
        add_q4(f"{name}.o", t(a + "o_proj.weight"))

    def mlp(name, p):
        w.add(f"{name}.ffn_norm", one_plus(p + "post_attention_layernorm.weight"), "f32", [hd])
        add_q4(f"{name}.gate_up", np.concatenate([t(p + "mlp.gate_proj.weight"), t(p + "mlp.up_proj.weight")]))
        add_q4(f"{name}.down", t(p + "mlp.down_proj.weight"))

    for i, kind in enumerate(c["layer_types"]):
        p = f"{pre}layers.{i}."
        print(f"layer {i} {kind}", flush=True)
        w.add(f"l{i}.attn_norm", one_plus(p + "input_layernorm.weight"), "f32", [hd])
        if kind == "linear_attention":
            a = p + "linear_attn."
            # one GEMV for all input projections: [q k v (conv input) | z (gate) | b (beta) | a (decay)]
            add_q4(f"l{i}.in_proj", np.concatenate([t(a + "in_proj_qkv.weight"), t(a + "in_proj_z.weight"),
                                                    t(a + "in_proj_b.weight"), t(a + "in_proj_a.weight")]))
            conv = t(a + "conv1d.weight")
            w.add(f"l{i}.conv_w", conv.reshape(conv.shape[0], -1), "f32", [conv.shape[0], conv.shape[2]])
            nh = c["linear_num_value_heads"]
            w.add(f"l{i}.neg_a", -np.exp(t(a + "A_log")), "f32", [nh])
            w.add(f"l{i}.dt_bias", t(a + "dt_bias"), "f32", [nh])
            w.add(f"l{i}.gate_norm", t(a + "norm.weight"), "f32", [c["linear_value_head_dim"]])
            add_q4(f"l{i}.o", t(a + "out_proj.weight"))
        else:
            full_attention(f"l{i}", p)
        mlp(f"l{i}", p)
    w.add("norm", one_plus(pre + "norm.weight"), "f32", [hd])

    print("mtp", flush=True)
    full_attention("mtp", "mtp.layers.0.")
    mlp("mtp", "mtp.layers.0.")
    w.add("mtp.attn_norm", one_plus("mtp.layers.0.input_layernorm.weight"), "f32", [hd])
    w.add("mtp.norm_e", one_plus("mtp.pre_fc_norm_embedding.weight"), "f32", [hd])
    w.add("mtp.norm_h", one_plus("mtp.pre_fc_norm_hidden.weight"), "f32", [hd])
    w.add("mtp.norm", one_plus("mtp.norm.weight"), "f32", [hd])
    add_q4("mtp.fc", t("mtp.fc.weight"))
    assert cfg.get("tie_word_embeddings", c.get("tie_word_embeddings"))

    tok = json.load(open(os.path.join(src, "tokenizer.json"), encoding="utf-8"))
    special = {x["content"]: x["id"] for x in tok["added_tokens"]}
    eos = c["eos_token_id"]
    eos = sorted({*(eos if isinstance(eos, list) else [eos]), special["<|endoftext|>"], special["<|im_end|>"]})
    rp = c["rope_parameters"]
    return {
        "hidden_size": hd,
        "intermediate_size": c["intermediate_size"],
        "num_layers": c["num_hidden_layers"],
        "num_heads": c["num_attention_heads"],
        "num_kv_heads": c["num_key_value_heads"],
        "head_dim": c["head_dim"],
        "vocab_size": c["vocab_size"],
        "rms_norm_eps": c["rms_norm_eps"],
        "rope_theta": rp["rope_theta"],
        "rotary_dim": int(c["head_dim"] * rp.get("partial_rotary_factor", 1.0)),
        "max_position_embeddings": c["max_position_embeddings"],
        "bos_token_id": None,
        "eos_token_ids": eos,
        "tie_word_embeddings": True,
        "layer_types": ["linear" if k == "linear_attention" else "full" for k in c["layer_types"]],
        "linear_num_heads": c["linear_num_value_heads"],
        "linear_key_head_dim": c["linear_key_head_dim"],
        "linear_value_head_dim": c["linear_value_head_dim"],
        "linear_conv_kernel": c["linear_conv_kernel_dim"],
        "mtp_layers": c.get("mtp_num_hidden_layers", 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="HF model directory")
    ap.add_argument("dst", help="output directory")
    ap.add_argument("--group", type=int, default=32)
    ap.add_argument("--search", action="store_true", help="per-group clip search for RTN tensors")
    ap.add_argument("--prequant", help="directory of GPTQ results (gptq.py); RTN for tensors not in it")
    ap.add_argument("--draft-vocab", help="draft_vocab.py ranking: MTP draft lm_head over the most frequent tokens")
    ap.add_argument("--draft-size", type=int, default=32768)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.src, "config.json")))
    arch = {"LlamaForCausalLM": "llama", "Qwen3_5ForConditionalGeneration": "qwen3_5"}[cfg["architectures"][0]]
    assert cfg.get("rope_scaling") is None, "rope_scaling not supported yet"

    st = {}
    for fn in sorted(os.listdir(args.src)):
        if fn.endswith(".safetensors"):
            st.update(load_safetensors(os.path.join(args.src, fn)))

    os.makedirs(args.dst, exist_ok=True)
    w = ShardWriter(args.dst)

    def t(name):
        return to_f32(st[name])

    def add_q4(name, mat):
        w.add_q4(name, mat, args.group, args.search, args.prequant)

    print("embed")
    add_q4("embed", t("model.language_model.embed_tokens.weight" if arch == "qwen3_5" else "model.embed_tokens.weight"))
    config = write_qwen35(w, cfg, t, add_q4, args.src) if arch == "qwen3_5" else write_llama(w, cfg, t, add_q4)
    if args.draft_vocab and config.get("mtp_layers"):
        # token ids of the draft lm_head; the engine gathers their embedding rows at load time
        ids = np.sort(np.array(json.load(open(args.draft_vocab))["ranked"][: args.draft_size], dtype=np.uint32))
        w.add("mtp.head_ids", ids, "u32", [len(ids)])
    n_shards = w.close()

    manifest = {
        "format": "webyiq-v2",
        "arch": arch,
        "config": config,
        "quant": {"bits": 4, "group": args.group, "scheme": "asym-gptq" if args.prequant else ("asym-rtn-search" if args.search else "asym-rtn")},
        "shards": [f"shard-{i:03d}.bin" for i in range(n_shards)],
        "tensors": w.tensors,
    }
    manifest["shard_hashes"] = shard_hashes(args.dst, manifest["shards"])
    json.dump(manifest, open(os.path.join(args.dst, "manifest.json"), "w"), indent=1)
    for fn in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        src = os.path.join(args.src, fn)
        if os.path.exists(src):
            with open(src, "rb") as a, open(os.path.join(args.dst, fn), "wb") as b:
                b.write(a.read())
    total = sum(v["bytes"] for v in w.tensors.values())
    print(f"done: {n_shards} shards, {total / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()

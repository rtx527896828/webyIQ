"""Torch reference for the Qwen3.5 MTP (multi-token prediction) head, which transformers ignores.

MTP at position i takes the main model's hidden state h_i and the embedding of token i+1 and
predicts token i+2:
  x = fc(cat(norm_e(embed(t_{i+1})), norm_h(h_i)))  ->  full-attention decoder layer  ->  norm  ->  lm_head
Its attention runs over its own KV of positions 0..i.

Measures greedy draft acceptance on greedy continuations of a few prompts, for the possible
conventions (hidden before / after the final norm, concat order).

  uv run python mtp_ref.py ../models/hf/Qwen3.5-2B [--gen 128]
"""

import argparse
import glob
import json
import os

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm

from gptq import positions, text_model

PROMPTS = [
    ("chat", "Explain how a hash map works and when you would use one instead of a sorted array."),
    ("chat", "用三段话介绍一下长城的历史。"),
    ("chat", "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring."),
    ("text", "The history of the Roman Empire spans more than a thousand years. It began"),
]


class MTP(torch.nn.Module):
    def __init__(self, cfg, model_dir):
        super().__init__()
        full = cfg.layer_types.index("full_attention")
        self.layer = Qwen3_5DecoderLayer(cfg, full)
        h = cfg.hidden_size
        self.fc = torch.nn.Linear(2 * h, h, bias=False)
        self.norm = Qwen3_5RMSNorm(h, cfg.rms_norm_eps)
        self.norm_e = Qwen3_5RMSNorm(h, cfg.rms_norm_eps)
        self.norm_h = Qwen3_5RMSNorm(h, cfg.rms_norm_eps)
        names = {"fc.weight": "mtp.fc.weight", "norm.weight": "mtp.norm.weight",
                 "norm_e.weight": "mtp.pre_fc_norm_embedding.weight", "norm_h.weight": "mtp.pre_fc_norm_hidden.weight"}
        sd = {}
        for f in glob.glob(os.path.join(model_dir, "*.safetensors")):
            with safe_open(f, "pt") as st:
                for k in st.keys():
                    if k.startswith("mtp.layers.0."):
                        sd["layer." + k[len("mtp.layers.0."):]] = st.get_tensor(k).float()
                    elif k in names.values():
                        sd[next(a for a, b in names.items() if b == k)] = st.get_tensor(k).float()
        missing, unexpected = self.load_state_dict(sd, strict=False)
        assert not unexpected and not missing, (missing, unexpected)

    def forward(self, emb, hid, pos_emb, order="eh"):
        e, h = self.norm_e(emb), self.norm_h(hid)
        x = self.fc(torch.cat([e, h] if order == "eh" else [h, e], -1))
        x = self.layer(x, attention_mask=None, position_embeddings=pos_emb)
        return self.norm(x[0] if isinstance(x, tuple) else x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--vocab", type=int, nargs="*", default=[16384, 32768, 65536], help="draft vocab sizes to try")
    ap.add_argument("--ranking", help="draft_vocab.py output; default: the first k token ids")
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count())

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    tm = text_model(model)
    mtp = MTP(tm.config, args.model).eval()

    stats = {}
    ranked = torch.tensor(json.load(open(args.ranking))["ranked"]) if args.ranking else None
    with torch.no_grad():
        for kind, p in PROMPTS:
            if kind == "chat":
                p = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids
            n0 = ids.shape[1]
            seq = model.generate(ids, max_new_tokens=args.gen, do_sample=False)
            n = seq.shape[1]
            # one forward over the whole greedy sequence; h_i for every position
            h = tm.embed_tokens(seq)
            pe = positions(tm, h)
            for layer in tm.layers:
                h = layer(h, attention_mask=None, position_embeddings=pe)
                h = h[0] if isinstance(h, tuple) else h
            # MTP at i uses t_{i+1} (all known, the sequence is greedy-consistent), predicts t_{i+2}
            emb = tm.embed_tokens(seq[:, 1:])
            for hk, hid in [("pre-norm", h[:, :-1]), ("post-norm", tm.norm(h)[:, :-1])]:
                for order in ["eh", "he"]:
                    out = mtp(emb, hid, positions(tm, emb), order)
                    logits = model.lm_head(out)[0]
                    for k in [None] + (args.vocab if (hk, order) == ("post-norm", "eh") else []):
                        # draft[i] predicts t_{i+2}; with k, the draft is restricted to k tokens
                        if k is None:
                            draft = logits.argmax(-1)
                        else:
                            sub = ranked[:k] if ranked is not None else torch.arange(k)
                            draft = sub[logits[:, sub].argmax(-1)]
                        hit = (draft[n0 - 1 : n - 2] == seq[0, n0 + 1 : n]).float()
                        s = stats.setdefault((hk, order, k), [0, 0])
                        s[0] += hit.sum().item()
                        s[1] += hit.numel()
            print(f"[{kind}] {tok.decode(seq[0, n0:])[:120]!r}", flush=True)
    for (hk, order, k), (a, t) in stats.items():
        print(f"hidden {hk:9} concat {order} draft vocab {k or 'all'}: acceptance {a / t:.3f} ({int(a)}/{t})")


if __name__ == "__main__":
    main()

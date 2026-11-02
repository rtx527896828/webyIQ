"""Perplexity on WikiText-2 (test), the quality metric for quantization / compression work.

  ppl.py data <hf_model_dir> <out.json>        tokenize -> {"windows": [[ids...], ...]}
  ppl.py ref  <hf_model_dir> <windows.json> [--fakequant] [--actquant]

Windows are independent (no carried context), so engine and reference see identical inputs.
The engine side is `webyiq-cli <model> --ppl <windows.json>`.
"""

import argparse
import io
import json
import math
import urllib.request

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from convert import quantize_q4

WIKITEXT = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet"
C4 = "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-validation.00000-of-00008.json.gz"


def cmd_data(args):
    import pyarrow.parquet as pq

    if args.source == "c4":
        import gzip

        docs = [json.loads(l)["text"] for l in gzip.decompress(urllib.request.urlopen(C4).read()).splitlines()]
        text = "\n\n".join(docs[:2000])
    else:
        raw = urllib.request.urlopen(WIKITEXT).read()
        text = "".join(pq.read_table(io.BytesIO(raw)).column("text").to_pylist())
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    w = args.window
    windows = [ids[i * w : (i + 1) * w] for i in range(args.n)]
    json.dump({"source": args.source, "window": w, "windows": windows}, open(args.out, "w"))
    print(f"{len(ids)} tokens total, wrote {args.n} windows of {w}")


def cmd_ref(args):
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    if args.fakequant:
        with torch.no_grad():
            for _, p in model.named_parameters():
                if p.dim() == 2:
                    p.copy_(torch.from_numpy(quantize_q4(p.numpy(), 32)[2]))
    if args.actquant:
        def fq_input(_mod, inputs):
            x = inputs[0]
            g = x.reshape(*x.shape[:-1], -1, 32)
            dx = g.abs().amax(-1, keepdim=True) / 127.0
            q = torch.where(dx > 0, torch.round(g / torch.where(dx > 0, dx, 1.0)), 0.0)
            return (q * dx).reshape(x.shape),

        for mod in model.modules():
            if isinstance(mod, torch.nn.Linear):
                mod.register_forward_pre_hook(fq_input)

    data = json.load(open(args.windows))
    nll, n = 0.0, 0
    with torch.no_grad():
        for w in data["windows"]:
            ids = torch.tensor([w])
            logp = torch.log_softmax(model(ids).logits[0, :-1].double(), -1)
            nll -= logp.gather(1, ids[0, 1:, None]).sum().item()
            n += len(w) - 1
    print(json.dumps({"fakequant": args.fakequant, "actquant": args.actquant, "tokens": n, "ppl": math.exp(nll / n)}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("model")
    d.add_argument("out")
    d.add_argument("--window", type=int, default=512)
    d.add_argument("--n", type=int, default=8)
    d.add_argument("--source", choices=["wikitext", "c4"], default="wikitext")
    r = sub.add_parser("ref")
    r.add_argument("model")
    r.add_argument("windows")
    r.add_argument("--fakequant", action="store_true")
    r.add_argument("--actquant", action="store_true")
    args = ap.parse_args()
    {"data": cmd_data, "ref": cmd_ref}[args.cmd](args)


if __name__ == "__main__":
    main()

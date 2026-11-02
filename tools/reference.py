"""Produce reference logits with HF transformers (CPU, fp32) for engine verification.

  --fakequant  apply the same int4 RTN quantization as convert.py to every linear
               weight + embedding.
  --actquant   also quantize every linear input to int8 (symmetric, groups of 32), as the
               engine's GEMV/GEMM do; with both flags the engine should match to float tolerance.
"""

import argparse
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from convert import quantize_q4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--chat", action="store_true", help="wrap prompt with the chat template")
    ap.add_argument("--fakequant", action="store_true")
    ap.add_argument("--actquant", action="store_true")
    ap.add_argument("--group", type=int, default=32)
    ap.add_argument("--gen", type=int, default=16)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()

    if args.fakequant:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.dim() == 2:
                    _, _, deq = quantize_q4(p.numpy(), args.group)
                    p.copy_(torch.from_numpy(deq))

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

    if args.chat:
        text = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    else:
        text = args.prompt
    ids = tok(text, add_special_tokens=False)["input_ids"]

    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0].numpy()
        gen = model.generate(torch.tensor([ids]), max_new_tokens=args.gen, do_sample=False)[0].tolist()[len(ids):]

    json.dump(
        {
            "prompt_ids": ids,
            "last_logits_top10": np.argsort(-logits[-1])[:10].tolist(),
            "greedy": gen,
            "greedy_text": tok.decode(gen),
        },
        open(args.out + ".json", "w"),
        indent=1,
    )
    logits.astype(np.float32).tofile(args.out + ".logits.f32")
    print(json.dumps({"n": len(ids), "greedy_text": tok.decode(gen)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

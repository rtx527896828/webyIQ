"""Tokenize chat prompts for the native runner (no tokenizer in Rust): writes {"prompts": [[ids...], ...]}.

  uv run python prompt_ids.py <hf_model_dir> <out.json> ["question" ...]
"""

import json
import sys

from transformers import AutoTokenizer

DEFAULT = [
    "Explain how a hash map works and when you would use one instead of a sorted array.",
    "用三段话介绍一下长城的历史。",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "What are the main differences between TCP and UDP? Answer with a short table.",
]

tok = AutoTokenizer.from_pretrained(sys.argv[1])
questions = sys.argv[3:] or DEFAULT
prompts = []
for q in questions:
    text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompts.append(tok(text, add_special_tokens=False)["input_ids"])
json.dump({"prompts": prompts}, open(sys.argv[2], "w"))
print([len(p) for p in prompts])

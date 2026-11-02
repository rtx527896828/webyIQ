"""Greedy-decode the exported model with onnxruntime (CPU EP), using the KV cache.

Usage: uv run python validate.py <export_dir> [n_new]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

d = Path(sys.argv[1])
n_new = int(sys.argv[2]) if len(sys.argv) > 2 else 8
cfg = json.loads((d / "config.json").read_text())
L, kvh, hd = cfg["num_hidden_layers"], cfg["num_key_value_heads"], cfg["head_dim"]

sess = ort.InferenceSession(str(d / "onnx" / "model_q4f16.onnx"), providers=["CPUExecutionProvider"])
names = {i.name for i in sess.get_inputs()}
tok = Tokenizer.from_file(str(d / "tokenizer.json"))

prompt = [608, 4894, 304, 6918, 357]  # "The capital of France is" (no BOS)
past = {f"past_key_values.{i}.{k}": np.zeros((1, kvh, 0, hd), np.float16) for i in range(L) for k in ("key", "value")}
ids, out = list(prompt), []
cur = prompt
t0 = time.time()
for step in range(n_new):
    feed = {"input_ids": np.array([cur], np.int64), "attention_mask": np.ones((1, len(ids)), np.int64), **past}
    if "position_ids" in names:
        feed["position_ids"] = np.arange(len(ids) - len(cur), len(ids), dtype=np.int64)[None]
    res = sess.run(None, feed)
    r = dict(zip([o.name for o in sess.get_outputs()], res))
    logits = r["logits"][0, -1]
    nxt = int(np.argmax(logits))
    if step == 0:
        top = np.argsort(-logits)[:5]
        print("step0 logits dtype", r["logits"].dtype, "top5", [(int(t), round(float(logits[t]), 3), tok.decode([int(t)])) for t in top])
    out.append(nxt)
    ids.append(nxt)
    cur = [nxt]
    past = {k.replace("present", "past_key_values"): v for k, v in r.items() if k.startswith("present")}
print("generated ids:", out)
print("text:", repr(tok.decode(out)))
print(f"{time.time() - t0:.1f}s")

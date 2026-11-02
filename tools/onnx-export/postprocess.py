"""Turn an onnxruntime-genai builder output into the transformers.js layout.

- casts fp16 logits to fp32 (like onnx-community q4f16 exports),
- pins the symbolic KV head-size dim (`kv_cache_dim`) to head_dim,
- writes onnx/model_q4f16.onnx + onnx/model_q4f16.onnx_data (single external data file),
- copies HF config/tokenizer files and adds `transformers.js_config` to config.json.

Usage: uv run python postprocess.py <builder_out_dir> <hf_dir> <dst_dir>
"""

import json
import shutil
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper

src, hf, dst = map(Path, sys.argv[1:4])
cfg = json.loads((hf / "config.json").read_text())
head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]

m = onnx.load(str(src / "model.onnx"), load_external_data=True)
g = m.graph

# 1) fp16 logits -> fp32
logits = next(o for o in g.output if o.name == "logits")
if logits.type.tensor_type.elem_type == TensorProto.FLOAT16:
    for n in g.node:
        for i, o in enumerate(n.output):
            if o == "logits":
                n.output[i] = "logits_fp16"
        for i, x in enumerate(n.input):
            if x == "logits":
                n.input[i] = "logits_fp16"
    g.node.append(helper.make_node("Cast", ["logits_fp16"], ["logits"], name="/lm_head/CastToFloat32", to=TensorProto.FLOAT))
    logits.type.tensor_type.elem_type = TensorProto.FLOAT

# 2) pin KV head dim
for v in list(g.input) + list(g.output):
    if v.name.startswith(("past_key_values.", "present.")):
        d = v.type.tensor_type.shape.dim[3]
        if d.dim_param:
            d.Clear()
            d.dim_value = head_dim

# 3) save with transformers.js file names
(dst / "onnx").mkdir(parents=True, exist_ok=True)
out = dst / "onnx" / "model_q4f16.onnx"
for p in (out, out.with_name(out.name + "_data")):
    p.unlink(missing_ok=True)
onnx.save_model(m, str(out), save_as_external_data=True, all_tensors_to_one_file=True,
                location="model_q4f16.onnx_data", size_threshold=1024)

# 4) HF side files
for f in ["generation_config.json", "tokenizer.json", "tokenizer_config.json",
          "special_tokens_map.json", "chat_template.jinja"]:
    if (hf / f).exists():
        shutil.copy2(hf / f, dst / f)
cfg["transformers.js_config"] = {
    "use_external_data_format": {"model_q4f16.onnx": 1},
    "kv_cache_dtype": {"q4f16": "float16", "fp16": "float16"},
}
(dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
print("wrote", dst)

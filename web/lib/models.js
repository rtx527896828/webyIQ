// Models offered by the demo pages. `dir`: webyIQ converted model (also holds the tokenizer used by
// both engines), `onnx`: transformers.js baseline under models/onnx/.
export const MODELS = {
  "qwen3.5-2b": {
    name: "Qwen3.5-2B",
    blurb: "Gated DeltaNet + gated attention, MTP speculative decoding",
    dir: "../models/qwen3.5-2b-gptq/",
    onnx: "Qwen3.5-2B-ONNX",
  },
  "minicpm5-2b": {
    name: "MiniCPM5-2B",
    blurb: "Llama architecture, 42 layers",
    dir: "../models/minicpm5-2b-gptq/",
    onnx: "MiniCPM5-2B-ONNX",
  },
};

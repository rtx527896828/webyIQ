// Models offered by the demo pages. Served from the local dev server (tools/serve.mjs) the pages
// read ../models/ (`dir`: webyIQ conversion, which also holds the tokenizer; `onnx`: transformers.js
// baseline under models/onnx/); anywhere else (the hosted demo) they load from the Hugging Face Hub.
// ?hub / ?local force either.
export const MODELS = {
  "qwen3.5-2b": {
    name: "Qwen3.5-2B",
    blurb: "Gated DeltaNet + gated attention, MTP speculative decoding",
    dir: "../models/qwen3.5-2b-gptq/",
    hub: "tixiren/Qwen3.5-2B-webyIQ",
    onnx: "Qwen3.5-2B-ONNX",
    onnxHub: "onnx-community/Qwen3.5-2B-ONNX",
  },
  "minicpm5-2b": {
    name: "MiniCPM5-2B",
    blurb: "Llama architecture, 42 layers",
    dir: "../models/minicpm5-2b-gptq/",
    hub: "tixiren/MiniCPM5-2B-webyIQ",
    onnx: "MiniCPM5-2B-ONNX",
    onnxHub: "tixiren/MiniCPM5-2B-ONNX",
  },
};

const q = new URLSearchParams(location.search);
export const HOSTED = q.has("hub") || (!q.has("local") && !["127.0.0.1", "localhost"].includes(location.hostname));

/** Base URL of a model's webyIQ files. */
export const modelDir = (m) => (HOSTED ? `https://huggingface.co/${m.hub}/resolve/main/` : m.dir);

/** transformers.js model id (local id under models/onnx/, or a Hub repo when hosted). */
export const onnxId = (m) => (HOSTED ? m.onnxHub : m.onnx);

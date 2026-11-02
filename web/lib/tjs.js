// transformers.js baseline (ONNX q4f16 on WebGPU) with the same generate() contract as webyiq.js.
import { AutoModelForCausalLM, InterruptableStoppingCriteria, Tensor, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1";

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = new URL("../models/onnx/", location.href).href;

/** Load an ONNX model from models/onnx/<id>. onProgress({ phase, bytes, total }). */
export async function loadTjs(id, { eos, onProgress = () => {} } = {}) {
  const t0 = performance.now();
  const files = new Map();
  const model = await AutoModelForCausalLM.from_pretrained(id, {
    dtype: "q4f16",
    device: "webgpu",
    progress_callback: (p) => {
      if (p.status === "progress" && p.total) {
        files.set(p.file, [p.loaded, p.total]);
        let bytes = 0, total = 0;
        for (const [l, t] of files.values()) (bytes += l), (total += t);
        onProgress({ phase: "weights", bytes, total });
      }
    },
  });
  onProgress({ phase: "warmup" });
  // warm-up: the first run compiles the WebGPU kernels
  await runGenerate(model, [1000, 1001, 1002, 1003], { maxTokens: 2, eos });
  return {
    model,
    loadMs: performance.now() - t0,
    generate: (ids, opts) => runGenerate(model, ids, { eos, ...opts }),
    dispose: () => model.dispose(),
  };
}

async function runGenerate(model, ids, { maxTokens = 512, onToken = () => {}, stop = () => false, signal = null, eos } = {}) {
  const input_ids = new Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]);
  const attention_mask = new Tensor("int64", new BigInt64Array(ids.length).fill(1n), [1, ids.length]);
  const interrupt = new InterruptableStoppingCriteria();
  const eosSet = new Set(eos ?? []);
  const out = [];
  let first = true, ttftMs = 0;
  const t0 = performance.now();
  const stats = () => {
    const now = performance.now();
    return { ttftMs, tokens: out.length, elapsedMs: now - t0, decodeTokS: out.length > 1 ? (out.length - 1) / (now - t0 - ttftMs) * 1000 : 0 };
  };
  const streamer = {
    put(v) {
      if (first) return void (first = false); // the first put() carries the prompt
      const t = Number(v[0][0]);
      if (eosSet.has(t)) return;
      if (!out.length) ttftMs = performance.now() - t0;
      out.push(t);
      onToken(out, stats());
      if (stop(out) || signal?.aborted) interrupt.interrupt();
    },
    end() {},
  };
  const onAbort = () => interrupt.interrupt();
  signal?.addEventListener("abort", onAbort);
  try {
    await model.generate({
      input_ids, attention_mask, max_new_tokens: maxTokens, do_sample: false, streamer,
      stopping_criteria: [interrupt], eos_token_id: eos,
    });
  } finally {
    signal?.removeEventListener("abort", onAbort);
  }
  const s = stats();
  return { ids: out, promptTokens: ids.length, ttftMs, decodeMs: s.elapsedMs - ttftMs, decodeTokS: s.decodeTokS };
}

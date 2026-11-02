// webyIQ in the browser: load a converted model (weights cached in the Cache API) and generate.
import init, { WebEngine } from "../pkg/webyiq_web.js";
import { ChatTokenizer } from "./tokenizer.js";

let wasmReady = null;

/**
 * Files of a converted model: `dir` is a URL (relative to the page) or a FileSystemDirectoryHandle
 * (a folder the user picked). get(name) -> Response; `cacheKey` is set for URL sources only (local
 * folders are not copied into the Cache API).
 */
function modelFiles(dir) {
  if (typeof dir === "string") {
    const base = new URL(dir, location.href).href;
    return { get: (f) => fetch(base + f), cacheKey: (f) => base + f };
  }
  return {
    get: async (f) => {
      try {
        return new Response(await (await dir.getFileHandle(f)).getFile());
      } catch {
        return new Response(null, { status: 404 });
      }
    },
    cacheKey: null,
  };
}

/** Tokenizer of a converted model directory (same files as the HF model). */
export async function loadTokenizer(dir) {
  const files = modelFiles(dir);
  const [tj, tc, tmpl] = await Promise.all(
    ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"].map((f, i) =>
      files.get(f).then((r) => (i < 2 ? r.json() : r.ok ? r.text() : null))));
  return new ChatTokenizer(tj, tc, tmpl);
}

/** Chat messages -> prompt token ids (thinking off unless asked). */
export function chatIds(tokenizer, messages, thinking = false) {
  const text = tokenizer.apply_chat_template(messages, { add_generation_prompt: true, enable_thinking: thinking });
  return tokenizer.encode(text, { add_special_tokens: false });
}

/**
 * Load a webyIQ model. onProgress({ phase, done, total, bytes }) during the download.
 * Returns { engine, tokenizer, manifest, adapter, loadMs, cached, spec, generate(ids, opts), dispose() }.
 */
export async function loadWebyiq(dir, { ctx = 4096, tokenizer = null, onProgress = () => {} } = {}) {
  if (!navigator.gpu) throw new Error("WebGPU is not available in this browser");
  wasmReady ??= init();
  await wasmReady;
  const files = modelFiles(dir);
  const res = await files.get("manifest.json");
  if (!res.ok) throw new Error(`no manifest.json in the model source (${res.status})`);
  const manifest = await res.json();
  const t0 = performance.now();
  const packedInt8 = navigator.gpu.wgslLanguageFeatures?.has("packed_4x8_integer_dot_product") ?? false;
  const engine = await WebEngine.create(JSON.stringify(manifest), ctx, true, packedInt8, true);

  // shards: Cache API keyed by URL + content hash, 3 downloads in flight, uploaded as they land
  const cache = files.cacheKey && (await caches.open("webyiq-weights").catch(() => null));
  const total = manifest.shards.length;
  let done = 0, bytes = 0, cached = 0, next = 0;
  onProgress({ phase: "weights", done, total, bytes });
  const fetchShard = async (i) => {
    const name = manifest.shards[i];
    const url = files.cacheKey?.(name);
    const key = url && manifest.shard_hashes ? `${url}?h=${manifest.shard_hashes[i]}` : url;
    let r = cache && (await cache.match(key));
    if (r) cached++;
    else {
      r = await files.get(name);
      if (!r.ok) throw new Error(`${name}: ${r.status}`);
      if (cache) await cache.put(key, r.clone()).catch(() => {});
    }
    const buf = new Uint8Array(await r.arrayBuffer());
    bytes += buf.byteLength;
    engine.upload_shard(i, buf);
    onProgress({ phase: "weights", done: ++done, total, bytes });
  };
  await Promise.all(Array.from({ length: 3 }, async () => {
    while (next < total) await fetchShard(next++);
  }));
  tokenizer ??= await loadTokenizer(dir);
  onProgress({ phase: "warmup", done, total, bytes });
  const warm = manifest.config.bos_token_id ?? manifest.config.eos_token_ids[0];
  await engine.prefill(new Uint32Array([warm]));
  engine.reset();
  const spec = engine.has_speculative ? await specCheck(engine) : null;
  engine.set_speculative(!!spec?.on);
  const loadMs = performance.now() - t0;
  const eos = new Set(manifest.config.eos_token_ids);

  /**
   * Greedy generation from prompt ids. opts: { maxTokens, onToken(ids, stats), stop(ids) -> bool, signal }.
   * Returns { ids, promptTokens, ttftMs, decodeMs, decodeTokS, steps }.
   */
  async function generate(ids, { maxTokens = 512, onToken = () => {}, stop = () => false, signal = null } = {}) {
    if (ids.length + maxTokens > engine.max_ctx) maxTokens = Math.max(1, engine.max_ctx - ids.length - 2);
    engine.reset();
    const t0 = performance.now();
    let tok = await engine.prefill(new Uint32Array(ids));
    const ttftMs = performance.now() - t0;
    const out = [];
    const stats = () => {
      const now = performance.now();
      return { ttftMs, tokens: out.length, elapsedMs: now - t0, decodeTokS: out.length > 1 ? (out.length - 1) / (now - t0 - ttftMs) * 1000 : 0 };
    };
    while (!eos.has(tok) && out.length < maxTokens && !signal?.aborted) {
      out.push(tok);
      onToken(out, stats());
      if (stop(out)) break;
      tok = await engine.next_token(4);
    }
    engine.discard_ahead();
    const s = stats();
    const [steps] = engine.spec_stats();
    return { ids: out, promptTokens: ids.length, ttftMs, decodeMs: s.elapsedMs - ttftMs, decodeTokS: s.decodeTokS, steps };
  }

  const info = (await navigator.gpu.requestAdapter({ powerPreference: "high-performance" }))?.info;
  return {
    engine, tokenizer, manifest, loadMs, cached, bytes, spec, generate,
    adapter: engine.adapter_info,
    gpuName: info?.description || [info?.vendor, info?.architecture].filter(Boolean).join(" ") || "WebGPU",
    dispose() {
      engine.free();
    },
  };
}

// Speculative steps cost the same whatever the acceptance, so whether they pay off on this GPU
// follows from ms per plain token vs ms per speculative step, with the draft acceptance measured
// on chat prompts (~1.68 tokens per step for Qwen3.5-2B; see README).
const SPEC_TOKENS_PER_STEP = 1.68;
async function specCheck(engine, n = 24) {
  const prompt = new Uint32Array(32).map((_, i) => 1000 + i * 37);
  const timed = async (spec) => {
    engine.set_speculative(spec);
    engine.reset();
    await engine.prefill(prompt);
    await engine.next_token(4);
    const s0 = engine.spec_stats()[0];
    const t = performance.now();
    for (let i = 0; i < n; i++) await engine.next_token(4);
    const ms = performance.now() - t;
    const steps = engine.spec_stats()[0] - s0;
    engine.discard_ahead();
    return spec ? ms / steps : ms / n;
  };
  const plain = await timed(false), step = await timed(true);
  engine.reset();
  const gain = SPEC_TOKENS_PER_STEP * plain / step;
  return { on: gain > 1.05, expected: gain };
}

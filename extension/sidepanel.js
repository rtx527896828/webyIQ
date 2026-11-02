// Side panel: loads the model (WebGPU, in this extension page) and runs the agent on a tab. The page
// is read and operated through chrome.scripting: lib/dom-agent.classic.js (lib/dom-agent.js wrapped
// as a plain script by tools/build-extension.mjs) is injected into the tab, then called.
import { loadWebyiq, loadTokenizer } from "./lib/webyiq.js";
import { runAgent } from "./lib/agent-core.js";

const $ = (id) => document.getElementById(id);
const q = new URLSearchParams(location.search);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
// automated tests report progress to the dev server
const testLog = q.has("autotest") ? (m) => fetch("http://127.0.0.1:8765/log?m=" + encodeURIComponent(m)).catch(() => {}) : () => {};

const EXAMPLES = [
  "Summarize what this page is about in one sentence.",
  "Search this site for “WebGPU” and tell me the title of the first result.",
  "Find the most recent item on this page and tell me its title.",
];

// ---- settings (source choice and picked folder survive panel restarts) --------------------
function idb(op, key, value) {
  return new Promise((resolve, reject) => {
    const open = indexedDB.open("webyiq-agent", 1);
    open.onupgradeneeded = () => open.result.createObjectStore("kv");
    open.onsuccess = () => {
      const tx = open.result.transaction("kv", op === "get" ? "readonly" : "readwrite");
      const r = op === "get" ? tx.objectStore("kv").get(key) : tx.objectStore("kv").put(value, key);
      r.onsuccess = () => resolve(r.result);
      r.onerror = () => reject(r.error);
    };
    open.onerror = () => reject(open.error);
  });
}

const settings = await chrome.storage.local.get({ source: $("source").value, url: "" });
$("source").value = q.get("source") ?? settings.source;
$("url").value = settings.url;
function showSource() {
  $("url").hidden = $("source").value !== "custom";
  $("folderhint").hidden = $("source").value !== "folder";
}
showSource();
$("source").onchange = () => {
  showSource();
  chrome.storage.local.set({ source: $("source").value });
};
$("url").onchange = () => chrome.storage.local.set({ url: $("url").value.trim() });

// ---- model ----------------------------------------------------------------------------------
let backend = null, tokenizer = null;

function status(text, cls = "") {
  $("status").textContent = text;
  $("status").className = "pill " + cls;
}

async function modelSource() {
  const v = $("source").value;
  if (v.startsWith("url:")) return v.slice(4);
  if (v === "custom") {
    const u = $("url").value.trim();
    if (!u) throw new Error("enter the URL of a converted model directory");
    return u.endsWith("/") ? u : u + "/";
  }
  let dir = await idb("get", "folder").catch(() => null);
  if (dir && (await dir.requestPermission({ mode: "read" })) !== "granted") dir = null;
  if (!dir) {
    dir = await showDirectoryPicker({ id: "webyiq-model", mode: "read" });
    await idb("put", "folder", dir).catch(() => {});
  }
  return dir;
}

async function load() {
  $("load").disabled = $("run").disabled = true;
  backend?.dispose();
  backend = null;
  status("loading", "run");
  const bar = $("bar");
  bar.hidden = false;
  try {
    const src = await modelSource();
    tokenizer = await loadTokenizer(src);
    backend = await loadWebyiq(src, {
      tokenizer,
      onProgress: (p) => {
        $("info").textContent = p.phase === "weights" ? `weights ${p.done}/${p.total} · ${(p.bytes / 2 ** 30).toFixed(2)} GiB` : "compiling kernels…";
        bar.querySelector("i").style.width = `${Math.round((p.phase === "weights" ? p.done / p.total : 1) * 100)}%`;
      },
    });
    const name = backend.manifest.arch === "qwen3_5" ? "Qwen3.5" : "MiniCPM";
    $("info").textContent = `${name} on ${backend.gpuName} · ready in ${(backend.loadMs / 1000).toFixed(1)} s` +
      (backend.cached ? ` (${backend.cached} shards from cache)` : "") + (backend.spec?.on ? " · speculative decoding on" : "");
    status("ready", "ok");
    $("run").disabled = false;
  } catch (e) {
    status("failed", "err");
    $("info").textContent = `Could not load: ${e.message || e}`;
  } finally {
    bar.hidden = true;
    $("load").disabled = false;
  }
}

// ---- the tab --------------------------------------------------------------------------------
const myWindow = (await chrome.windows.getCurrent()).id;
let testTab = null; // ?tab=URL: the agent drives its own window (automated tests)

async function targetTab() {
  if (testTab) return chrome.tabs.get(testTab);
  const [tab] = await chrome.tabs.query({ active: true, windowId: myWindow });
  return tab;
}

async function showTab() {
  const tab = await targetTab().catch(() => null);
  $("tabinfo").textContent = tab ? `Tab: ${tab.title || tab.url}` : "";
}
chrome.tabs.onActivated.addListener(showTab);
chrome.tabs.onUpdated.addListener((id, change) => change.title && showTab());
showTab();

// runs in the page (the extension's isolated world), after dom-agent.classic.js
function inPage(op, arg) {
  const m = globalThis.__webyiqDom;
  return op === "observe" ? m.observePage(document) : m.actOnPage(document, arg);
}

// executeScript with a time limit (a page that never settles must not hang the agent)
function inject(details, what) {
  return Promise.race([
    chrome.scripting.executeScript(details),
    sleep(what === "act" ? 6000 : 15000).then(() => {
      throw new Error(`the page did not respond (${what})`);
    }),
  ]);
}

async function callPage(tabId, op, arg = null) {
  await inject({ target: { tabId }, files: ["lib/dom-agent.classic.js"], injectImmediately: true }, "load");
  testLog(`${op}: injected`);
  const [r] = await inject({ target: { tabId }, func: inPage, args: [op, arg], injectImmediately: true }, op);
  testLog(`${op}: done`);
  return r?.result;
}

// wait for a navigation the action may have started (or a script-driven page update)
async function settle(tabId) {
  const t0 = Date.now();
  await sleep(250);
  let tab = await chrome.tabs.get(tabId);
  while (tab.status !== "complete" && Date.now() - t0 < 20000) {
    await sleep(150);
    tab = await chrome.tabs.get(tabId);
  }
  await sleep(tab.status === "complete" && Date.now() - t0 > 400 ? 300 : 500);
  return chrome.tabs.get(tabId);
}

function tabEnv(tabId) {
  const after = async (result, urlBefore) => {
    const tab = await settle(tabId);
    return tab.url !== urlBefore ? `${result}, opened "${(tab.title || tab.url).slice(0, 80)}"` : result;
  };
  return {
    async observe() {
      await settle(tabId);
      testLog("observe");
      return callPage(tabId, "observe");
    },
    async act(a) {
      const before = (await chrome.tabs.get(tabId)).url;
      let result;
      try {
        result = await callPage(tabId, "act", { name: a.name, args: a.args });
      } catch (e) {
        // the action navigated away while the injected script was still running: that is success
        if ((await chrome.tabs.get(tabId)).url === before) throw e;
      }
      return after(result ?? `${a.src} done`, before);
    },
    async back() {
      const before = (await chrome.tabs.get(tabId)).url;
      await chrome.tabs.goBack(tabId).catch(() => {});
      const tab = await settle(tabId);
      return tab.url !== before ? `opened "${tab.title}"` : "no previous page";
    },
  };
}

// ---- agent ----------------------------------------------------------------------------------
let abort = null;

function stepCard(n) {
  const d = document.createElement("div");
  d.className = "step";
  d.innerHTML = `<div class="t"><b>Step ${n}</b><span>thinking…</span></div><div class="thought"></div><span class="act"></span><div class="res"></div>`;
  $("steps").append(d);
  d.scrollIntoView({ block: "nearest" });
  return d;
}

async function run() {
  const goal = $("goal").value.trim();
  if (!goal || !backend) return null;
  const tab = await targetTab();
  if (!tab || !/^(https?|file):/.test(tab.url ?? "")) {
    $("tabinfo").textContent = "This tab cannot be automated (browser and store pages are off limits). Open a web page.";
    return null;
  }
  abort = new AbortController();
  $("run").disabled = true;
  $("stop").disabled = false;
  $("answer").hidden = true;
  $("steps").innerHTML = "";
  status("running", "run");
  let card = null;
  let r;
  try {
    r = await runAgent({
      goal, env: tabEnv(tab.id), llm: backend, tokenizer, maxSteps: +q.get("steps") || 12, signal: abort.signal,
      onEvent: (e) => {
        if (e.type === "step") {
          card = stepCard(e.n);
          testLog(`step ${e.n}`);
        }
        else if (e.type === "thought") card.querySelector(".thought").textContent = e.text;
        else if (e.type === "done") {
          const s = e.step;
          card.querySelector(".t span").textContent = `${s.promptTokens} tok · ${(s.ms / 1000).toFixed(2)} s`;
          card.querySelector(".thought").textContent = s.thought || "(no thought)";
          card.querySelector(".act").textContent = s.action ?? "unreadable reply";
          card.classList.toggle("bad", !s.action);
          testLog(`step ${e.n}: ${s.action} -> ${s.result}`);
          if (s.result !== "answer") card.querySelector(".res").textContent = "→ " + s.result;
        }
      },
    });
    $("answer-text").textContent = r.answer ?? (abort.signal.aborted ? "Stopped." : "No answer within the step limit.");
    $("answer-stats").textContent = `${r.steps.length} steps · model ${(r.modelMs / 1000).toFixed(1)} s · total ${(r.totalMs / 1000).toFixed(1)} s`;
    status("ready", "ok");
  } catch (e) {
    $("answer-text").textContent = `Error: ${e.message || e}`;
    $("answer-stats").textContent = "";
    status("error", "err");
  }
  $("answer").hidden = false;
  $("answer").scrollIntoView({ block: "nearest" });
  $("run").disabled = false;
  $("stop").disabled = true;
  return r;
}

for (const ex of EXAMPLES) {
  const b = document.createElement("button");
  b.textContent = ex;
  b.onclick = () => ($("goal").value = ex);
  $("examples").append(b);
}
$("load").onclick = load;
$("run").onclick = run;
$("stop").onclick = () => abort?.abort();
$("goal").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) run();
});

// test hook: sidepanel.html?autotest&tab=<url>&goal=<text>[&source=url:<model dir>&steps=N]; the
// agent drives a new window and the result goes to the dev server
if (q.has("autotest")) {
  const log = testLog;
  log("panel started");
  (async () => {
    const win = await chrome.windows.create({ url: q.get("tab"), focused: true });
    testTab = win.tabs[0].id;
    log("tab opened");
    await load();
    log(`loaded: ${$("info").textContent}`);
    $("goal").value = q.get("goal");
    const r = await run();
    await fetch("http://127.0.0.1:8765/report", { method: "POST", body: JSON.stringify({ agenttest: { model: backend?.manifest.arch, backend: "extension", gpu: backend?.gpuName, ...r } }) });
  })().catch((e) => fetch("http://127.0.0.1:8765/report", { method: "POST", body: JSON.stringify({ error: String(e.stack || e) }) }));
}

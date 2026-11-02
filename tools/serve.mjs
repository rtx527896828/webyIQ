// Dev server: static files from the repo root, POST /report (appends JSON lines to
// models/reports.jsonl) so benchmark pages on other browsers/devices can send results back, and
// GET /proxy?url=... for the browser-agent demo (web/agent.html).
//   node tools/serve.mjs [--port 8765] [--host 127.0.0.1]
//
// The proxy serves pages of a few public sites from this origin so that the demo page can read
// and operate their DOM (a cross-origin iframe is off limits). GET only, HTML only, scripts and
// inline handlers stripped, a <base> element added so styles and images load from the site itself.
import { createReadStream, appendFileSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join, normalize, resolve } from "node:path";

const args = process.argv.slice(2);
const opt = (name, def) => (args.includes(name) ? args[args.indexOf(name) + 1] : def);
const port = +opt("--port", 8765);
const host = opt("--host", "127.0.0.1");
const root = resolve(import.meta.dirname, "..");
const reports = join(root, "models", "reports.jsonl");
const types = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css",
  ".wasm": "application/wasm", ".json": "application/json", ".jinja": "text/plain; charset=utf-8", ".svg": "image/svg+xml",
  ".bin": "application/octet-stream", ".onnx": "application/octet-stream", ".onnx_data": "application/octet-stream",
};

const PROXY_HOSTS = new Set(["en.wikipedia.org", "zh.wikipedia.org", "simple.wikipedia.org", "news.ycombinator.com"]);
const proxyCache = new Map(); // url -> { html, final }, small LRU

function sanitize(html, base) {
  return html
    .replace(/<script\b[\s\S]*?<\/script\s*>/gi, "")
    .replace(/<meta[^>]+http-equiv=["']?content-security-policy[^>]*>/gi, "")
    .replace(/\son[a-z]+\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)/gi, "")
    .replace(/<head([^>]*)>/i, `<head$1><base href="${base.replace(/"/g, "&quot;")}">`);
}

async function proxy(target, res) {
  let u;
  try {
    u = new URL(target);
  } catch {
    return res.writeHead(400).end("bad url");
  }
  if (u.protocol !== "https:" || !PROXY_HOSTS.has(u.hostname)) {
    res.writeHead(403, { "content-type": "text/html; charset=utf-8" });
    return res.end(`<p style="font:16px system-ui;padding:2em">This demo only proxies ${[...PROXY_HOSTS].join(", ")}.<br>${u.hostname} is not on the list.</p>`);
  }
  let hit = proxyCache.get(u.href);
  if (!hit) {
    const r = await fetch(u.href, {
      headers: { "user-agent": "webyIQ-demo/0.1 (local single-user browser-agent demo)", "accept": "text/html" },
      redirect: "follow",
      signal: AbortSignal.timeout(15000),
    });
    if (!(r.headers.get("content-type") ?? "").includes("text/html")) return res.writeHead(415).end("not html");
    hit = { final: r.url, status: r.status, html: sanitize(await r.text(), r.url) };
    proxyCache.set(u.href, hit);
    if (proxyCache.size > 200) proxyCache.delete(proxyCache.keys().next().value);
  }
  res.writeHead(hit.status, { "content-type": "text/html; charset=utf-8", "x-final-url": hit.final, "cache-control": "no-store" });
  res.end(hit.html);
}

createServer((req, res) => {
  const url = new URL(req.url, "http://x");
  if (req.method === "POST" && url.pathname === "/report") {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => {
      const line = JSON.stringify({ time: new Date().toISOString(), ua: req.headers["user-agent"], ...JSON.parse(body) });
      appendFileSync(reports, line + "\n");
      console.log("report:", line.slice(0, 300));
      res.writeHead(204).end();
    });
    return;
  }
  if (url.pathname === "/log") {
    // progress lines from test pages (e.g. the extension's side panel)
    console.log("log:", url.searchParams.get("m"));
    return res.writeHead(204, { "access-control-allow-origin": "*" }).end();
  }
  if (req.method === "GET" && url.pathname === "/proxy") {
    proxy(url.searchParams.get("url") ?? "", res).catch((e) => {
      if (!res.headersSent) res.writeHead(502, { "content-type": "text/plain" });
      res.end(String(e));
    });
    return;
  }
  const path = normalize(join(root, decodeURIComponent(url.pathname)));
  if (!path.startsWith(root)) return res.writeHead(403).end();
  let st;
  try {
    st = statSync(path);
  } catch {
    return res.writeHead(404).end();
  }
  if (st.isDirectory()) return res.writeHead(404).end();
  res.writeHead(200, { "content-type": types[extname(path)] ?? "application/octet-stream", "content-length": st.size });
  createReadStream(path).pipe(res);
}).listen(port, host, () => console.log(`serving ${root} on http://${host}:${port}`));

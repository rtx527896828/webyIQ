// Assemble the static demo site (site/) for hosting, e.g. on GitHub Pages:
//   node tools/build-site.mjs        (after wasm-pack has built web/pkg)
// Chat arena (also the index page) and the browser-agent page; on a static host the pages load the
// models from the Hugging Face Hub (see web/lib/models.js). The benchmark pages stay local tools.
import { copyFileSync, mkdirSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const site = join(root, "site");
rmSync(site, { recursive: true, force: true });
const copy = (from, to) => {
  mkdirSync(join(site, to, ".."), { recursive: true });
  copyFileSync(join(root, from), join(site, to));
};

copy("web/chat.html", "index.html");
copy("web/chat.html", "chat.html");
copy("web/agent.html", "agent.html");
for (const dir of ["lib", "vendor"]) for (const f of readdirSync(join(root, "web", dir))) copy(`web/${dir}/${f}`, `${dir}/${f}`);
for (const f of ["webyiq_web.js", "webyiq_web_bg.wasm"]) copy(`web/pkg/${f}`, `pkg/${f}`);
writeFileSync(join(site, ".nojekyll"), ""); // serve files as they are

console.log(`site assembled in ${site}`);

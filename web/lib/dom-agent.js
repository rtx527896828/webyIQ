// DOM side of the browser agent: describe a page as numbered interactive elements + text, and
// perform actions on it. Works on any document: the demo page passes its frame's document, the
// browser extension injects it into the tab (chrome.scripting) and runs it on `document`; it must
// stay import-free, since tools/build-extension.mjs wraps it into a plain script for that.

const clean = (s) => (s ?? "").replace(/\s+/g, " ").trim();
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export function visible(el) {
  if (!el.getClientRects().length) return false;
  const s = el.ownerDocument.defaultView.getComputedStyle(el);
  return s.visibility !== "hidden" && s.display !== "none" && s.opacity !== "0";
}

// elements of the document and of open shadow roots (web components), in document order
function* allElements(root) {
  const walker = (root.ownerDocument ?? root).createTreeWalker(root, NodeFilter.SHOW_ELEMENT);
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    yield n;
    if (n.shadowRoot) yield* allElements(n.shadowRoot);
  }
}
const queryAll = (doc, selector) => [...allElements(doc)].filter((el) => el.matches(selector));

// el is region or a descendant of it, also across shadow roots
function inside(region, el) {
  for (let n = el; n; n = n.parentNode ?? n.host) if (n === region) return true;
  return false;
}

const BLOCKS = new Set(["p", "div", "li", "ul", "ol", "dl", "dt", "dd", "tr", "td", "th", "table", "section", "article", "header",
  "footer", "nav", "aside", "main", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "blockquote", "br", "figcaption", "summary"]);

// rendered text through open shadow roots and slots (innerText stops at shadow hosts)
function flatText(node, out) {
  if (node.nodeType === Node.TEXT_NODE) return void out.push(node.data);
  if (node.nodeType === Node.ELEMENT_NODE) {
    if (["script", "style", "noscript", "template"].includes(node.localName) || node.checkVisibility?.() === false) return;
    if (node.localName === "slot" && node.assignedNodes().length) {
      for (const n of node.assignedNodes({ flatten: true })) flatText(n, out);
      return;
    }
    if (node.shadowRoot) {
      flatText(node.shadowRoot, out);
      return void out.push(" ");
    }
  } else if (node.nodeType !== Node.DOCUMENT_FRAGMENT_NODE) {
    return;
  }
  for (const c of node.childNodes) flatText(c, out);
  if (BLOCKS.has(node.localName)) out.push(" ");
}

function regionText(el) {
  if (![...allElements(el)].some((e) => e.shadowRoot)) return clean(el.innerText);
  const out = [];
  flatText(el, out);
  return clean(out.join(""));
}

function mainRegion(doc) {
  for (const sel of ["#mw-content-text", "#hnmain", "main", "[role=main]", "article", "#content"]) {
    const el = doc.querySelector(sel);
    if (el && regionText(el).length > 200) return el;
  }
  return doc.body;
}

export function pageText(doc) {
  return regionText(mainRegion(doc) ?? doc.body).replace(/\[\s*(edit|编辑|编辑源代码|\d+)\s*\]/g, "");
}

function isTextInput(el) {
  const tag = el.tagName.toLowerCase();
  if (tag === "textarea" || el.isContentEditable) return true;
  return tag === "input" && !["submit", "button", "image", "reset", "checkbox", "radio", "file", "range", "color"].includes(el.type);
}

function describe(el) {
  const tag = el.tagName.toLowerCase();
  const label = (s) => `"${clean(s).slice(0, 70)}"`;
  const aria = el.getAttribute("aria-label") || el.title || "";
  if (tag === "a" && el.hasAttribute("href")) return `link ${label(el.innerText || aria)}`;
  if (isTextInput(el)) {
    const lab = el.placeholder || aria || el.name || el.id || "";
    const value = el.isContentEditable ? el.innerText : el.value;
    return `input ${label(lab)}${value ? ` value=${label(value)}` : ""}`;
  }
  if (tag === "select") return `select ${label(aria || el.name || "")} value=${label(el.selectedOptions?.[0]?.text ?? "")}`;
  return `button ${label(el.innerText || el.value || aria)}`;
}

const CANDIDATES = 'input:not([type=hidden]), textarea, select, button, a[href], [role=button], [role=link], [role=tab], [role=menuitem], [role=searchbox], [role=textbox], [contenteditable=true], summary';

/**
 * Snapshot of the page for the model: { title, url, elements: ["link \"...\"", ...], text }.
 * The element objects are kept on the document for the following `act` call.
 */
export function observePage(doc, { maxElements = 44 } = {}) {
  const main = mainRegion(doc);
  const seen = new Set();
  const controls = [], inMain = [], rest = [];
  for (const el of queryAll(doc, CANDIDATES)) {
    if (!visible(el)) continue;
    const tag = el.tagName.toLowerCase();
    if (tag === "a" && el.hasAttribute("href")) {
      const text = clean(el.innerText || el.getAttribute("aria-label") || el.title);
      const href = el.getAttribute("href");
      // skip in-page anchors, script links, edit links and links to pages that do not exist (wiki red links)
      if (text.length < 2 || href.startsWith("#") || href.startsWith("javascript:") || /^\[?(edit|编辑)/i.test(text)) continue;
      if (el.classList.contains("new") || /[?&](action=edit|redlink=1)/.test(href)) continue;
      const k = text + "|" + el.href;
      if (seen.has(k)) continue;
      seen.add(k);
      (inside(main, el) ? inMain : rest).push(el);
    } else if (tag === "input" && ["checkbox", "radio"].includes(el.type)) {
      continue;
    } else {
      controls.push(el);
    }
  }
  const els = [...controls.slice(0, 10), ...inMain.slice(0, 32), ...rest.slice(0, Math.max(0, 40 - inMain.length))].slice(0, maxElements);
  doc.__webyiqElements = els;
  let text = pageText(doc);
  // after a jump to #fragment, the text starts at the target (e.g. one entry of a long reference page)
  const hash = doc.defaultView.location.hash;
  if (hash.length > 1) {
    const head = clean(doc.getElementById(decodeURIComponent(hash.slice(1)))?.innerText).slice(0, 40);
    const at = head ? text.indexOf(head) : -1;
    if (at > 0) text = text.slice(at);
  }
  return { title: clean(doc.title), url: doc.baseURI, elements: els.map(describe), text };
}

// the page's search input: a text input with a search-like name (inside a form if possible; may be
// collapsed, as Wikipedia's header search when the site's scripts are off), else the first visible one
function searchBox(doc) {
  const inputs = queryAll(doc, 'input[type=search], input[type=text], input:not([type]), [role=searchbox], textarea');
  const named = inputs.filter((i) => i.getAttribute("role") === "searchbox" || i.type === "search" ||
    /search|query|^q$|keyword|搜索/i.test(`${i.name} ${i.id} ${i.placeholder ?? ""} ${i.getAttribute("aria-label") ?? ""}`));
  return named.find((i) => i.closest("#searchform")) ?? named.find((i) => visible(i) && i.form) ?? named.find(visible) ??
    named[0] ?? inputs.find(visible) ?? null;
}

// URL a GET form would navigate to (submitter: the clicked button, if any)
function formUrl(form, submitter) {
  const base = form.ownerDocument.baseURI;
  const action = new URL(form.getAttribute("action") || base, base);
  const data = new FormData(form, submitter ?? undefined);
  for (const [k, v] of data) if (typeof v === "string") action.searchParams.set(k, v);
  return action.href;
}

async function highlight(el) {
  if (!visible(el)) return;
  el.scrollIntoView({ block: "center", behavior: "smooth" });
  await sleep(250);
  const doc = el.ownerDocument;
  const r = el.getBoundingClientRect();
  const box = doc.createElement("div");
  box.style.cssText = `position:fixed;left:${r.left - 4}px;top:${r.top - 4}px;width:${r.width + 8}px;height:${r.height + 8}px;` +
    "border:3px solid #7c5cff;border-radius:8px;box-shadow:0 0 0 6px rgba(124,92,255,.25);z-index:2147483647;pointer-events:none;transition:opacity .4s";
  doc.documentElement.append(box);
  await sleep(350);
  box.style.opacity = "0";
  setTimeout(() => box.remove(), 500);
}

// set a value the way typing does, so frameworks (React, Vue) see it
function setValue(el, value) {
  if (el.isContentEditable) {
    el.focus();
    el.textContent = value;
  } else {
    const proto = Object.getPrototypeOf(el);
    const setter = Object.getOwnPropertyDescriptor(proto, "value")?.set;
    setter ? setter.call(el, value) : (el.value = value);
  }
  el.dispatchEvent(new Event("input", { bubbles: true }));
  el.dispatchEvent(new Event("change", { bubbles: true }));
}

function pressEnter(el) {
  for (const type of ["keydown", "keypress", "keyup"]) {
    el.dispatchEvent(new KeyboardEvent(type, { key: "Enter", code: "Enter", keyCode: 13, which: 13, bubbles: true, cancelable: true }));
  }
}

/**
 * Perform click / type / search on the page; returns a short result for the model.
 * `navigate(url)` (optional) handles link and GET-form navigation itself (the demo page routes it
 * through its proxy) and resolves to the new page's title; without it the page navigates natively.
 */
export async function actOnPage(doc, action, { navigate = null } = {}) {
  const { name, args } = action;
  const els = doc.__webyiqElements ?? [];
  const el = typeof args[0] === "number" ? els[args[0] - 1] : args[0]?.nodeType === 1 ? args[0] : null;
  const opened = (title) => (title ? `, opened "${title.slice(0, 80)}"` : "");
  switch (name) {
    case "click": {
      if (!el || !el.isConnected) return `there is no element [${args[0]}]`;
      if (isTextInput(el)) return `[${args[0]}] is an input: use type(${args[0]}, "words") to type into it`;
      await highlight(el);
      const tag = el.tagName.toLowerCase();
      if (navigate && tag === "a" && el.hasAttribute("href")) return `clicked${opened(await navigate(el.href))}`;
      // (a <button> in a form is type "submit" unless it says otherwise)
      const form = el.form ?? el.closest("form");
      if (navigate && form && ["submit", "image"].includes(el.type)) return `clicked${opened(await navigate(formUrl(form, el)))}`;
      el.click();
      return `clicked ${describe(el)}`;
    }
    case "type":
    case "search": {
      let input = name === "search" ? searchBox(doc) : el;
      if (name === "search" && !(input && visible(input))) {
        // search fields that only appear after pressing a "Search" button
        const button = queryAll(doc, "button, [role=button], a").find((b) => visible(b) &&
          /search|搜索/i.test(`${b.getAttribute("aria-label") ?? ""} ${b.title} ${clean(b.innerText)}`) && clean(b.innerText).length < 30);
        if (button) {
          button.click();
          await sleep(500);
          input = searchBox(doc) ?? input;
        }
      }
      const text = String(name === "search" ? args[0] ?? "" : args[1] ?? "");
      if (!input) return name === "search" ? "this page has no search box" : `there is no element [${args[0]}]`;
      if (!isTextInput(input)) return `[${args[0]}] is not an input`;
      await highlight(input);
      input.focus();
      for (let i = 1; i <= text.length; i++) {
        setValue(input, text.slice(0, i));
        await sleep(18);
      }
      await sleep(150);
      const form = input.form ?? input.closest("form");
      if (navigate && form) return `typed "${text}"${opened(await navigate(formUrl(form)))}`;
      pressEnter(input);
      if (form && input.isConnected) {
        // pages that do not handle Enter themselves
        await sleep(100);
        if (input.isConnected && doc.defaultView) form.requestSubmit?.();
      }
      return `typed "${text}" and pressed Enter`;
    }
    default:
      return `unknown action ${name}`;
  }
}

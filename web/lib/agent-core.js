// Model side of the browser agent: prompt, action parsing and the observe -> decide -> act loop.
// The page is reached through an `env` (see runAgent), so the demo page (proxied frame) and the
// browser extension (any tab) share all of this.
import { chatIds } from "./webyiq.js";

export const SYSTEM = `You are a web agent. You operate a real web page through a numbered list of its elements.
Each turn you get the goal, the current page and your previous steps. Reply with exactly two lines:
Thought: <one short sentence>
Action: <one action>

Actions:
search("topic")         search this site for a topic with its search box
click(7)                click element [7] (a link or button; use a real number from the list)
type(3, "some words")   type into input [3] and press Enter
find("some words")      show the sentences of this page that contain the words
scroll()                show the next part of the page text
back()                  go back to the previous page
answer("the answer")    finish with the answer to the goal

Rules:
- To look something up, use search("...") with the name of the topic only, e.g. search("Eiffel Tower"), not the
  question. search() finds and opens the site's search box itself: do not click search buttons.
- On a page of search results, click the result whose title is the topic.
- When the page text contains the answer, reply answer(...) at once. Answer exactly what the goal asks,
  briefly, in the language of the goal.
- A goal may need two lookups: first find the intermediate fact, then search for that.

Examples of a first step:
Goal: In which year was the Golden Gate Bridge opened?
Thought: The topic is the Golden Gate Bridge, so I look it up.
Action: search("Golden Gate Bridge")

Goal: 珠穆朗玛峰有多高？
Thought: 主题是珠穆朗玛峰，先搜索它。
Action: search("珠穆朗玛峰")`;

/** "Action: name(args)" -> { name, args, src } or null. */
export function parseAction(text) {
  // the action line up to its last ")" (arguments may contain parentheses)
  const m = text.match(/Action:\s*([a-z_]+)\s*\(([^\n]*)\)/i);
  if (!m) {
    // an answer(...) without the Action: label (possibly unquoted, at the end of a line)
    const bare = text.match(/\banswer\s*\(\s*"?([^\n]*?)"?\s*\)\s*$/im);
    return bare ? { name: "answer", args: [bare[1]], src: `answer("${bare[1]}")` } : null;
  }
  let args;
  try {
    args = JSON.parse(`[${m[2].replace(/'/g, '"')}]`);
  } catch {
    args = [m[2].replace(/^["']|["']$/g, "")];
  }
  return { name: m[1].toLowerCase(), args, src: `${m[1]}(${m[2]})` };
}

// the reply's Action: line is finished: it ends with ")" and its quotes and parentheses balance
// (so `answer("zip() returns ...")` does not stop at the first ")")
function actionComplete(text) {
  const line = text.match(/Action:([^\n]*)$/)?.[1];
  if (!line || !/\)\s*$/.test(line)) return /Action:[^\n]*\n/.test(text);
  const quotes = (line.match(/"/g) ?? []).length;
  const depth = (line.match(/\(/g) ?? []).length - (line.match(/\)/g) ?? []).length;
  return quotes % 2 === 0 && depth === 0;
}

function findSentences(text, words) {
  const terms = words.toLowerCase().split(/\s+/).filter(Boolean);
  const scored = text.split(/(?<=[.!?。！？])\s*/)
    .map((s) => [terms.filter((t) => s.toLowerCase().includes(t)).length, s])
    .filter(([n]) => n > 0)
    .sort((a, b) => b[0] - a[0])
    .slice(0, 4)
    .map(([, s]) => "- " + s.slice(0, 240));
  return scored.length ? `Sentences containing "${words}":\n${scored.join("\n")}` : `No sentence contains "${words}".`;
}

// one window of the page text; about the same number of tokens for CJK and latin text
function textWindow(text, part, chars) {
  const cjk = (text.slice(0, 2000).match(/[　-鿿]/g)?.length ?? 0) / Math.min(2000, text.length || 1);
  const win = Math.round(chars * (1 - 0.55 * cjk));
  const parts = Math.max(1, Math.ceil(text.length / win));
  part = Math.min(part, parts - 1);
  return { part, parts, window: text.slice(part * win, (part + 1) * win) };
}

function observation(goal, page, history, w, note) {
  let url = page.url;
  try {
    url = decodeURI(url);
  } catch {}
  const lines = [
    `Goal: ${goal}`,
    `Page: ${page.title} (${url})`,
    "",
    "Previous steps:",
    history.length ? history.map((h, i) => `${i + 1}. ${h.action} -> ${h.result}`).join("\n") : "(none)",
    "",
    "Elements:",
    page.elements.map((e, i) => `[${i + 1}] ${e}`).join("\n") || "(none)",
    "",
    `Page text (part ${w.part + 1} of ${w.parts}):`,
    w.window || "(empty)",
  ];
  if (note) lines.push("", note);
  return lines.join("\n");
}

/**
 * Run the agent until it answers, `maxSteps` pass or `signal` aborts.
 *   env: { observe() -> { title, url, elements: [description], text }, act(action) -> result, back() -> result }
 *   llm: { generate(ids, opts) } (webyiq.js / tjs.js), tokenizer: the model's chat tokenizer
 *   onEvent: { type: "step", n } | { type: "thought", n, text } | { type: "done", n, step, tokensIn, tokensOut, ms }
 * Returns { goal, answer, steps, modelMs, totalMs }.
 */
export async function runAgent({ goal, env, llm, tokenizer, maxSteps = 10, signal = null, onEvent = () => {}, textChars = 1800, debug = false }) {
  const history = [], steps = [];
  let part = 0, note = "", lastUrl = "", answer = null, modelMs = 0;
  const t0 = performance.now();
  const thoughtOf = (text) => (text.match(/Thought:\s*([^\n]*)/) ?? [, ""])[1];
  for (let n = 1; n <= maxSteps && !signal?.aborted; n++) {
    onEvent({ type: "step", n });
    const page = await env.observe();
    if (page.url !== lastUrl) part = 0;
    lastUrl = page.url;
    const w = textWindow(page.text, part, textChars);
    part = w.part;
    const obs = observation(goal, page, history, w, note);
    note = "";
    const ids = chatIds(tokenizer, [{ role: "system", content: SYSTEM }, { role: "user", content: obs }]);
    const ts = performance.now();
    const r = await llm.generate(ids, {
      maxTokens: 160,
      signal,
      onToken: (out) => onEvent({ type: "thought", n, text: thoughtOf(tokenizer.decode(out, { skip_special_tokens: true })) }),
      stop: (out) => actionComplete(tokenizer.decode(out, { skip_special_tokens: true })),
    });
    const ms = performance.now() - ts;
    modelMs += ms;
    const text = tokenizer.decode(r.ids, { skip_special_tokens: true });
    const a = parseAction(text);
    const step = { n, promptTokens: r.promptTokens, outTokens: r.ids.length, ms: Math.round(ms), thought: thoughtOf(text), action: a?.src ?? null, raw: a ? undefined : text.slice(0, 200) };
    if (debug) step.obs = obs;
    steps.push(step);
    if (!a) {
      step.result = "unreadable reply";
      history.push({ action: "(invalid reply)", result: "reply with a Thought: line and an Action: line", url: page.url });
      onEvent({ type: "done", n, step });
      continue;
    }
    if (a.name === "answer") {
      answer = String(a.args[0] ?? "");
      step.result = "answer";
      onEvent({ type: "done", n, step });
      break;
    }
    let result;
    if (a.name === "find") {
      note = findSentences(page.text, String(a.args[0] ?? ""));
      result = note.startsWith("No") ? "no match" : "matches shown below the page text";
    } else if (a.name === "scroll") {
      part++;
      result = `showing part ${Math.min(part, w.parts - 1) + 1} of the page text`;
    } else if (a.name === "back") {
      result = await env.back();
    } else {
      result = await env.act(a).catch((e) => `failed: ${e.message || e}`);
    }
    const again = history.some((h) => h.action === a.src && h.url === page.url);
    if (again) result += " (you already did exactly this, try something else)";
    step.result = result;
    history.push({ action: a.src, result, url: page.url });
    onEvent({ type: "done", n, step });
  }
  return { goal, answer, steps, modelMs: Math.round(modelMs), totalMs: Math.round(performance.now() - t0) };
}

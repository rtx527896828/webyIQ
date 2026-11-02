"""Pretty-print the last benchmark report(s) from models/reports.jsonl (written by tools/serve.mjs)."""

import json
import sys

n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
lines = open("models/reports.jsonl", encoding="utf-8").read().splitlines()[-n:]
for line in lines:
    d = json.loads(line)
    if "error" in d:
        print("error:", d["error"][:500])
        continue
    if "tjs" in d:
        for r in d["tjs"]:
            print(f"transformers.js {r['adapter']}: prompt {r['prompt_tokens']}: ttft {r['ttft_ms']} ms, "
                  f"prefill {r['prefill_tok_s']} tok/s, decode {r['decode_tok_s']} tok/s, load {r['load_ms']} ms")
        continue
    if "compare" in d:
        c = d["compare"]
        print("compare on", c["adapter"])
        for r in c["results"]:
            print(f"  {r['engine']:<42} chat: ttft {r['chat_ttft_ms']:>7} ms, decode {r['chat_decode_tok_s']:>6} tok/s | "
                  f"long ({r['long_prompt_tokens']} tok): ttft {r['long_ttft_ms']} ms ({r['long_prefill_tok_s']} tok/s) | load {r['load_ms']} ms")
            print("     ", " | ".join(x.replace(chr(10), " ")[:50] for x in r["samples"]))
        continue
    if "chattest" in d or "agenttest" in d:
        print(json.dumps(d.get("chattest") or d.get("agenttest"), ensure_ascii=False, indent=1)[:4000])
        continue
    if "probe" in d:
        print("probe:", json.dumps(d["probe"])[:500])
        continue
    print(d["load"]["adapter"], "|", d["load"].get("engine", ""), "| load", round(d["load"]["load_ms"]), "ms")
    if d["load"].get("spec_auto"):
        print("  speculative auto-check:", d["load"]["spec_auto"])
    for r in d["runs"]:
        print(f"  prompt {r['prompt_tokens']}: ttft {r['ttft_ms']} ms, prefill {r['prefill_tok_s']} tok/s, "
              f"decode {r['decode_tok_s']} tok/s (unpipelined {r['decode_unpipelined_tok_s']})")
    p = d.get("profile") or {}
    if p:
        print(f"  profile @pos {p['pos']}: {p['total_us']:.0f} us")
        for name, us, c in p["rows"][:8]:
            print(f"    {name:<14}{us:>8.0f} us  x{c}")
        if p.get("spec"):
            print(f"  speculative step profile: {sum(r[1] for r in p['spec']):.0f} us")
            for name, us, c in p["spec"][:8]:
                print(f"    {name:<14}{us:>8.0f} us  x{c}")
        if p.get("prefill128"):
            print(f"  prefill profile (128 tokens): {sum(r[1] for r in p['prefill128']):.0f} us")
            for name, us, c in p["prefill128"][:8]:
                print(f"    {name:<14}{us:>8.0f} us  x{c}")
    c = d.get("chatbench")
    if c:
        line = f"  chat bench ({c['prompts']} prompts, {c['tokens']} tokens): plain {c['plain_tok_s']} tok/s"
        if "spec_tok_s" in c:
            line += (f", speculative {c['spec_tok_s']} tok/s ({c['speedup']}x, {c['tokens_per_step']} tokens/step, "
                     f"{'identical' if c['identical'] else 'MISMATCH'})")
        print(line)

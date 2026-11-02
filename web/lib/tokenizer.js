// Chat tokenizer without transformers.js: @huggingface/tokenizers + @huggingface/jinja (vendored, so
// it also works where remote code is not allowed, as in the browser extension). Same calls as the
// transformers.js PreTrainedTokenizer used before: apply_chat_template, encode, decode.
import { Tokenizer } from "../vendor/tokenizers.min.mjs";
import { Template } from "../vendor/jinja.mjs";

export class ChatTokenizer {
  constructor(tokenizerJson, config, chatTemplate) {
    this.tok = new Tokenizer(tokenizerJson, config);
    this.config = config;
    this.template = new Template(chatTemplate ?? config.chat_template);
    // special tokens the templates refer to by name
    this.special = {};
    for (const k of ["bos_token", "eos_token", "pad_token", "unk_token"]) {
      const v = config[k];
      this.special[k] = v && typeof v === "object" ? v.content : v;
    }
  }

  /** Rendered prompt text (add_generation_prompt defaults to true). */
  apply_chat_template(messages, { add_generation_prompt = true, enable_thinking = false } = {}) {
    return this.template.render({ messages, add_generation_prompt, enable_thinking, ...this.special });
  }

  /** Token ids of `text` (no special tokens added unless asked; the chat template has them). */
  encode(text, { add_special_tokens = false } = {}) {
    return this.tok.encode(text, { add_special_tokens }).ids;
  }

  decode(ids, opts = {}) {
    return ids.length ? this.tok.decode(Array.from(ids), opts) : "";
  }
}

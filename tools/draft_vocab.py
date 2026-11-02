"""Rank the vocabulary for a reduced speculative-draft lm_head.

The MTP draft only needs to guess likely tokens, and every draft is verified against the full
vocabulary, so its lm_head can cover a frequent subset (a smaller GEMV per step). Token frequency
is counted on three equally weighted sources: English prose (WikiText-2 train), Chinese
Wikipedia, and Python code (the sources installed in this environment). Special tokens rank first.

  uv run python draft_vocab.py <hf_model_dir> <zhwiki.parquet> <out.json> [--tokens 2000000]

Writes {"ranked": [token ids, most frequent first]}; convert.py --draft-vocab takes the first N.
"""

import argparse
import glob
import io
import json
import os
import sysconfig
import urllib.request

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from gptq import WIKITEXT_TRAIN


def texts_wikitext():
    raw = urllib.request.urlopen(WIKITEXT_TRAIN).read()
    yield from (t for t in pq.read_table(io.BytesIO(raw)).column("text").to_pylist() if t.strip())


def texts_parquet(path):
    for batch in pq.ParquetFile(path).iter_batches(columns=["text"], batch_size=256):
        yield from batch.column("text").to_pylist()


def texts_python():
    for f in sorted(glob.glob(os.path.join(sysconfig.get_paths()["purelib"], "**", "*.py"), recursive=True)):
        with open(f, encoding="utf-8", errors="ignore") as h:
            yield h.read()


def counts(tok, texts, budget, vocab):
    c = np.zeros(vocab, dtype=np.int64)
    n = 0
    batch = []
    for t in texts:
        batch.append(t[:20000])
        if len(batch) == 64:
            for ids in tok(batch, add_special_tokens=False)["input_ids"]:
                np.add.at(c, ids, 1)
                n += len(ids)
            batch = []
            if n >= budget:
                break
    return c, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("zhwiki")
    ap.add_argument("out")
    ap.add_argument("--tokens", type=int, default=2_000_000, help="tokens counted per source")
    ap.add_argument("--keep", type=int, default=65536, help="length of the ranking written")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    vocab = json.load(open(os.path.join(args.model, "config.json")))
    vocab = vocab.get("text_config", vocab)["vocab_size"]
    score = np.zeros(vocab)
    for name, texts in [("wikitext", texts_wikitext()), ("zhwiki", texts_parquet(args.zhwiki)), ("python", texts_python())]:
        c, n = counts(tok, texts, args.tokens, vocab)
        print(f"{name}: {n} tokens, {np.count_nonzero(c)} distinct", flush=True)
        score += c / n

    special = sorted(tok.added_tokens_decoder)
    score[special] = np.inf
    ranked = np.argsort(-score, kind="stable")[: args.keep]
    seen = np.count_nonzero(score)
    for k in (8192, 16384, 32768, 65536):
        print(f"top {k}: covers {score[ranked[:k]][np.isfinite(score[ranked[:k]])].sum() / 3:.4f} of the counted tokens")
    print(f"{seen} tokens seen at all")
    json.dump({"ranked": ranked.tolist()}, open(args.out, "w"))


if __name__ == "__main__":
    main()

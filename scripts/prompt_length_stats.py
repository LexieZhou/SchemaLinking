"""Token length of the S1 / S2 schema serializations, per dataset.

Answers two questions before SFT:
- Does S2 fit in the training max_seq_length, or do we need S1 as a fallback?
- How many tokens does a full-schema prompt cost (the baseline side of the token comparison)?

Only the schema is counted. The full prompt adds a system message, the question, BIRD's evidence
and the target SQL on top, typically a few hundred tokens.

Stats are reported two ways:
- per database: every schema counted once.
- per example: each database weighted by its number of questions, which is what training sees.

All Qwen3 sizes share one tokenizer, so the 0.6B one (already cached locally) is used by default.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from sqlagent.schema import load_spider_tables, serialize_s1, serialize_s2

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
SNAILS_TABLES = RAW / "snails/SNAILS/db/schema-tables-json/spider_format"

# name -> (tables.json, questions json or None)
DATASETS = {
    "bird_train": (RAW / "bird/train/train_tables.json", RAW / "bird/train/train.json"),
    "bird_dev": (RAW / "bird/dev_20240627/dev_tables.json", RAW / "bird/dev_20240627/dev.json"),
    "snails_native": (SNAILS_TABLES / "snails_tables_Native.json", None),
}
THRESHOLDS = [2048, 4096, 8192]
SERIALIZERS = {"S1": serialize_s1, "S2": serialize_s2}


def describe(lengths: np.ndarray, weights: np.ndarray) -> str:
    """Weighted percentiles plus the share of mass above each threshold."""
    order = np.argsort(lengths)
    lengths, weights = lengths[order], weights[order]
    cdf = np.cumsum(weights) / weights.sum()

    def pct(q):
        return int(lengths[np.searchsorted(cdf, q)])

    over = "  ".join(
        f">{t // 1024}k={weights[lengths > t].sum() / weights.sum():6.1%}" for t in THRESHOLDS
    )
    return f"p50={pct(0.5):6d}  p90={pct(0.9):6d}  p99={pct(0.99):6d}  max={lengths[-1]:6d}  {over}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--top", type=int, default=5, help="list the N longest databases (by S2)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    for name, (tables_path, questions_path) in DATASETS.items():
        if not tables_path.exists():
            print(f"[{name}] skipped, missing {tables_path}")
            continue
        schemas = load_spider_tables(tables_path)
        db_ids = list(schemas)
        n_tokens = {
            s: np.array([len(tok(fn(schemas[d]))["input_ids"]) for d in db_ids])
            for s, fn in SERIALIZERS.items()
        }

        print(f"\n=== {name}: {len(db_ids)} databases")
        for s in SERIALIZERS:
            print(f"  {s} per database: {describe(n_tokens[s], np.ones(len(db_ids)))}")

        if questions_path is not None and questions_path.exists():
            with open(questions_path) as f:
                counts = Counter(ex["db_id"] for ex in json.load(f))
            weights = np.array([counts.get(d, 0) for d in db_ids], dtype=float)
            print(f"  ({int(weights.sum())} examples)")
            for s in SERIALIZERS:
                print(f"  {s} per example:  {describe(n_tokens[s], weights)}")

        ratio = n_tokens["S2"] / n_tokens["S1"]
        print(f"  S2/S1 ratio: median={np.median(ratio):.2f}  max={ratio.max():.2f}")

        print(f"  longest {args.top} (S1 / S2 tokens, #columns):")
        for i in np.argsort(-n_tokens["S2"])[: args.top]:
            d = db_ids[i]
            print(f"    {d:40s} {n_tokens['S1'][i]:6d} / {n_tokens['S2'][i]:6d}  "
                  f"{schemas[d].num_columns} cols")


if __name__ == "__main__":
    main()

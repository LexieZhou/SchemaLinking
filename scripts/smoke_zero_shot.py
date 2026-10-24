"""Phase 0 smoke test: zero-shot Qwen3 on a few BIRD dev questions, just to confirm the pipeline runs.

The real serializer will live in src/serialize.py (phase 1) and real inference in src/infer.py (vLLM).
"""
import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = (
    "You are a schema linker. Given a database schema and a question, output the tables "
    "and columns the SQL query would reference, as JSON mapping table names to column lists, "
    'e.g. {"table1": ["col1", "col2"], "table2": []}. Output only the JSON.'
)


def serialize_s1(db):
    cols = {i: [] for i in range(len(db["table_names_original"]))}
    for t, c in db["column_names_original"]:
        if t >= 0:
            cols[t].append(c)
    return "\n".join(f"{name}({', '.join(cols[i])})" for i, name in enumerate(db["table_names_original"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--questions", default="data/raw/bird/dev_20240627/dev.json")
    ap.add_argument("--tables", default="data/raw/bird/dev_20240627/dev_tables.json")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()

    dbs = {d["db_id"]: d for d in json.load(open(args.tables))}
    for ex in json.load(open(args.questions))[: args.n]:
        user = f"Schema:\n{serialize_s1(dbs[ex['db_id']])}\n\nQuestion: {ex['question']}"
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt").to(device)
        t0 = time.time()
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        text = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"--- [{ex['db_id']}] {ex['question']}")
        print(f"prompt_tokens={inputs['input_ids'].shape[1]}  latency={time.time() - t0:.1f}s")
        print("gold SQL:", ex["SQL"])
        print("model   :", text.strip())


if __name__ == "__main__":
    main()

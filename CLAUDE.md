# CLAUDE.md

Multi-turn SQL data agent on Qwen3-1.7B: the agent never sees the full schema, explores a SQLite
database through tools, then submits SQL. Pipeline: LoRA SFT → multi-turn GRPO (verl).
Core question: does reward = EX + λ·(schema-linking F1) beat EX alone, especially out of
distribution on SNAILS (obscure enterprise identifiers)?

`plan.md` is the source of truth for scope, design decisions and timeline. Read it before
starting a new phase, and update its checkboxes when a task is done.

## Hard rules

- **The GitHub repo is public.** Never commit course release-packet files (the course's
  `eval.py`, `sql_to_schema_links.py`, `schemas/`), any PDF, anything under `data/raw/`,
  model checkpoints, or API keys / `.env` files.
- **SNAILS is test-only.** Never train, tune λ or select checkpoints on it.
- **Split by database, never by example.** No db_id may appear in two splits.
- **Never drop examples silently.** Gold SQL that errors, times out or returns nothing is logged
  with a reason and excluded explicitly.
- **All database access is read-only** (`sqlagent.schema.connect_readonly`).
- **Loss only on model-generated tokens.** Tool outputs and prompts are masked, in both cold-start
  SFT and GRPO. Rerun the mask test after any chat-template change.

## Workflow

- One step = one branch `stepN-<short-name>` → PR → squash merge into `main`.
- Before committing, walk the user through every new file and design choice, and wait for their
  go-ahead. The user must be able to defend each decision in an interview.
- Code comments and docstrings in English. Match the density and idiom of the surrounding code.
- Every experiment is launched from a config file in `configs/`; no hyperparameters that exist
  only on a command line.

## Compute

- **Mac (M2, 8GB):** data pipeline, parsing, evaluation, environment and unit tests, tokenizer
  work, tiny smoke runs. No real training or batched inference.
- **RunPod A40 48GB:** SFT, vLLM evaluation, GRPO. Terminate pods when done. BIRD train databases
  and SNAILS sqlite databases are only downloaded there.

## Commands

```bash
conda activate schemalink
pip install -e .                 # once; editable install of src/sqlagent
pytest -q                        # all tests (data tests skip if data/raw is missing)
pytest -q -m "not data"          # unit tests only
bash scripts/download_data.sh    # raw data into data/raw/ (FULL_BIRD=1 on the GPU box)
```

## Layout

```
src/sqlagent/        library code; imported by scripts and tests
  schema.py          Schema model, loaders (tables.json / live sqlite), S1/S2 serializers
  parse_sql.py       SQL → schema links (sqlglot)                       (planned)
  executor.py        sandboxed sqlite: read-only, timeout, row cap      (planned)
  evaluate.py        EX + linking F1, per-example records               (planned)
  datasets.py        raw BIRD / Spider / SNAILS → unified examples      (planned)
  stats.py           paired bootstrap, McNemar                          (planned)
  prompts.py         system prompts and chat formatting                 (planned)
  env.py             tool environment (list_tables ... submit_sql)      (planned)
  reward.py          R = 1[format] · (R_ex + λ·R_link)                  (planned)
  agent.py           multi-turn rollout loop, tool-call parsing         (planned)
  trajectories.py    scripted cold-start trajectories                   (planned)
  inference.py       generation backends: HF (local), vLLM (GPU)        (planned)
scripts/             thin CLI entry points; logic lives in src/ so it can be tested
configs/{sft,grpo,eval}/   one YAML per experiment
tests/               pytest; mark tests that need data/raw with @pytest.mark.data
data/raw/            downloads (git-ignored)
data/processed/      unified jsonl, rebuilt by scripts (git-ignored)
outputs/<run_name>/  predictions, eval results, checkpoints (git-ignored)
```

`evaluate.py` is deliberately not named `eval.py`: it would shadow the builtin and be easy to
confuse with the course's file of that name.

## Data conventions

- Unified example (jsonl): `db_id`, `question`, `gold_sql`, `schema_links`, `db_path`, plus
  `source` and `split`.
- BIRD's `evidence` field is dropped (it names columns outright and SNAILS has none).
- Spider: `train_spider.json` only, not `train_others.json`.
- SNAILS: Native and Least variants only, using the sqlite-dialect gold SQL.
- SQLite identifiers are case-insensitive: look names up through `Schema.table()` /
  `Table.column()`, never with raw string equality.

## Agent and training defaults

- Qwen3 native `<tool_call>{json}</tool_call>` format, `enable_thinking=False`.
- Tools: `list_tables`, `describe_table`, `search_columns`, `sample_rows`, `execute_sql`,
  `submit_sql`. At most 8 turns; tool output truncated to ~1k tokens; `execute_sql` returns
  ≤20 rows with a 5 s timeout.
- Debug every pipeline on Qwen3-0.6B before running 1.7B.

# SQL Data Agent — Project Plan

Last updated: 2026-10-24

## 1. Overview

Over roughly four weeks, train a multi-turn **SQL data agent** on Qwen3-1.7B. The agent never sees the full schema. It explores a SQLite database through tools, then submits a SQL query. Training pipeline: LoRA SFT → multi-turn GRPO (verl).

The project answers one concrete question: **in multi-turn RL for a small model, does adding a dense schema-linking reward beat an execution-only reward, especially on real enterprise schemas with obscure identifiers?**

### Research questions

1. **Baseline.** How well does a single-turn LoRA SFT model that sees the full schema in its prompt score on BIRD dev (in-distribution) and SNAILS (out-of-distribution), measured by execution accuracy (EX)? How does it compare with a few-shot frontier API on accuracy, latency and cost?
2. **Agent vs. single-turn.** How much does a multi-turn tool-using agent (cold-start SFT → GRPO) improve over single-turn SFT? How many input tokens does it save on large schemas?
3. **Core.** Does reward = EX + λ·(schema-linking F1) converge faster and lose less out of distribution than EX alone?
4. **Behavior.** After RL, does the agent learn to explore more when the schema is larger or its names are more obscure?

### Deliverables

- Public GitHub repo: data pipeline, tool environment, training and evaluation scripts, reproducible configs
- Results tables + a W&B report (every number traceable to a run)
- Technical blog post: method, core comparison, behavior analysis, failure cases
- A one-line summary of the result, with every number taken from real experiments

### Target headline result (fill in brackets with real numbers)

> Built a multi-turn SQL data agent (Qwen3-1.7B) that explores databases via tools; trained with LoRA SFT cold-start → multi-turn GRPO in verl on a sandboxed SQLite environment. A dense schema-linking reward improved execution accuracy by [X] pts over execution-only reward; on enterprise schemas with obfuscated identifiers (SNAILS) the agent dropped [Y] pts vs [Z] for a full-schema SFT baseline, at [W]× fewer input tokens.

### What changed from the original plan

- The task moved from "output schema links in one turn" to "output SQL over multiple turns". Schema linking now serves as a reward and a diagnostic metric.
- SFT ablations cut from 6 to 2. The QLoRA, LoRA-rank and S3 (sample values) ablations are dropped.
- SNAILS is evaluated on two variants only: Native and Least.

## 2. Related work and positioning

Multi-turn RL for SQL agents already has solid prior work. This project does not aim to be first. It uses SkyRL-SQL as the reference implementation and extends it in three directions that prior work has not yet covered.

| Work | What it does | Model size | How we use it |
| --- | --- | --- | --- |
| [SkyRL-SQL](https://novasky-ai.notion.site/skyrl-sql) (NeurIPS 2025 workshop) | Multi-turn RL: query the DB → read execution feedback → revise the SQL; beats GPT-4o with ~600 training examples | 7B | **Reference implementation**: built on verl, code and data are open |
| [TRUST-SQL](https://arxiv.org/abs/2603.16448) (Mar 2026) | No schema in the prompt; Explore → Propose → Generate → Confirm; SFT warm-up + Phase-Aware GRPO | 4B / 8B / 14B | Closest design; borrow its exploration protocol and credit assignment |
| [SQL-Trail](https://aclanthology.org/2026.acl-long.1677/) (ACL 2026) | Adaptive turn budget + composite reward (correctness + exploration efficiency) | — | Borrow reward design |
| [AutoLink](https://arxiv.org/abs/2511.17190) (AAAI 2026) | Schema linking as agentic exploration; high recall even with 3,000+ columns | Large-model prompting | Shows exploratory linking works on large schemas |

**What sets this project apart** (no prior work found, but only a few searches were run; re-check before writing the blog post):

1. **Sub-2B.** All of the RL work above uses models of 4B or more.
2. **Dense schema-linking reward.** Prior work mostly uses EX as the main reward. Linking F1 appears only as an optional part of a composite reward in [Graph-Reward-SQL](https://aclanthology.org/2025.findings-emnlp.694.pdf) and [PaVeRL-SQL](https://arxiv.org/pdf/2509.07159).
3. **SNAILS naturalness evaluation.** No RL or agent work has been evaluated on [SNAILS](https://rmarcus.info/dbscholar/papers/7069). The original paper found that small models lose up to 20% execution accuracy on obscure identifiers.

## 3. Tech stack and compute

Main model Qwen3-1.7B, Qwen3-0.6B for debugging. SFT with TRL + PEFT, RL with verl. CPU work on the Mac; SFT, evaluation and GRPO on a rented RunPod A40. Total budget about 60–80 GPU-hours, roughly $40–50.

| Component | Choice | Rationale |
| --- | --- | --- |
| Main model | Qwen3-1.7B | Sub-2B sets us apart from TRUST-SQL (4B+); native tool calling; add a 4B run if budget allows |
| Debug model | Qwen3-0.6B | Get every pipeline working on it first |
| Thinking mode | Off by default (`enable_thinking=False`) | Multi-turn trajectories are already long; turning it on is an optional ablation |
| SFT | TRL `SFTTrainer` + PEFT LoRA | One codebase for the single-turn baseline and cold-start |
| RL | verl multi-turn tool-use rollout | Same framework as SkyRL-SQL, so the experience transfers directly |
| Inference | vLLM | Shared by evaluation and rollout |
| SQL parsing | `sqlglot` | Extract schema links from SQL, for the reward and diagnostics |
| Execution sandbox | Python `sqlite3`, read-only + timeout + row cap | Milliseconds per step, reproducible |
| Experiment tracking | Weights & Biases | Every run traceable |

### Compute

| Resource | Spec | Use | Status |
| --- | --- | --- | --- |
| Mac (local) | M2, 8GB RAM | Parser, evaluation, data statistics, environment unit tests | Ready |
| DSMLP Independent Study | GTX 1080 Ti 11GB | No vLLM or bf16 support; backup only | Available |
| RunPod A40 | 48GB, Secure Cloud $0.59/hr | Single-turn SFT, cold-start SFT, evaluation, GRPO (training, reference model and vLLM share one GPU during GRPO) | Rent on demand; confirm price before creating |

GPU-hour estimate: single-turn SFT ~10 h, cold-start + evaluation ~8 h, multi-turn GRPO (2 seeds) ~35 h, times a 1.3–1.5 waste factor. RunPod prices checked on 2026-10-24.

## 4. Task and environment design

Each episode starts with only the question and the db_id; the agent cannot see the schema. It may call tools for up to 8 turns and must end with `submit_sql`. The reward is given only at the last step.

### Tools

Qwen3's native `<tool_call>{json}</tool_call>` format; each tool output is truncated to about 1k tokens.

| Tool | Returns | Purpose |
| --- | --- | --- |
| `list_tables()` | All table names + row counts | Entry point for exploration |
| `describe_table(table)` | Column names, types, primary keys, foreign keys | Fetch schema details on demand |
| `search_columns(keyword)` | Fuzzy lexical match over all column names; top-10 `table.column` | Shortcut for large schemas; fails on SNAILS' obscure names, which is exactly what we want to test |
| `sample_rows(table, k≤5)` | First k rows, values truncated | Read values to infer meaning, e.g. guess that column `Abd` means abundance |
| `execute_sql(sql)` | Read-only execution; first 20 rows + total row count, or the error message; 5 s timeout | Trial runs and self-correction |
| `submit_sql(sql)` | Ends the episode | Final answer |

### Reward

```
R = 1[format ok] · (R_ex + λ · R_link)
```

- **R_ex ∈ {0, 1}**: whether the predicted and gold SQL return the same result. Results are compared as multisets; order matters only when the gold SQL has ORDER BY.
- **R_link ∈ [0, 1]**: `parse_sql.py` extracts schema links from both the predicted and gold SQL; score = 0.5 × table-level mean(P, R, F1) + 0.5 × column-level mean(P, R, F1). A query with wrong logic still earns partial credit if it found the right tables and columns.
- **Format**: no submit within 8 turns, or invalid tool-call JSON → R = 0.
- **Variants compared**: **A** R_ex only; **B** R_ex + 0.3·R_link (add λ = 0.1 if budget allows); **C** (optional) R_link only, to check whether linking alone is insufficient.
- **Anti-hacking**: precision is part of R_link, so adding irrelevant columns is penalized; λ < 1 keeps EX the dominant signal. Monitor average turns and the average number of tables and columns referenced during training.

### Loss mask

This is the most common source of bugs in agentic RL. Tool outputs are not generated by the model, so both cold-start SFT and GRPO compute loss only on model-generated tokens. Otherwise the model learns to imitate database output instead of learning to call tools. Write a unit test that prints the mask and check it token by token.

### Single-turn baseline

Same model, with the full schema in the prompt (S1 compact format, or S2 with types and PK/FK), generating SQL directly without tools.

## 5. Data

Training uses only BIRD + Spider; SNAILS is test-only. All splits are by database, never by example. Every example uses one jsonl format: `db_id`, `question`, `gold_sql`, `schema_links`, `db_path`.

| Split | Source | Size | Use |
| --- | --- | --- | --- |
| train | BIRD train + Spider train, holding out ~10% of databases | ~15k | Single-turn SFT, cold-start trajectories, GRPO pool |
| val | Held-out databases | ~1.5k | Checkpoint selection, tuning λ |
| GRPO pool | Filtered from train: sample the cold-start model 8 times per example, drop all-correct and all-wrong | ~3–5k | GRPO training |
| test-ID | BIRD dev (2024-06-27 revision) | 1534 examples, 11 DBs | In-distribution main result |
| test-OOD | SNAILS Native + Least, using the sqlite SQL and databases | 503 each, 17 DBs | Generalization and naming robustness |
| Backup OOD | SNAILS' renamed Spider dev (12 DBs) | TBD | Fallback if SNAILS shows a floor effect |

### Design decisions

- **Drop BIRD's `evidence` field.** It often names columns outright, and SNAILS has no such field. Training with it could teach the model to copy from evidence, which would not transfer OOD. The cost is that BIRD dev scores will be lower than leaderboard numbers; say so in the report.
- **Spider: use only `train_spider.json` (7,000 examples)**, not `train_others.json`.
- **Cold-start trajectories are generated from gold by a script**: `list_tables` → `describe_table` on the gold tables (with 1 random irrelevant exploration step inserted, so the model does not learn that every step is always correct) → `execute_sql(gold)` → `submit_sql`. No API calls needed.

### Databases to download on the GPU machine

- [ ] BIRD train databases: `FULL_BIRD=1 bash scripts/download_data.sh` (~9GB download, 30GB+ unzipped)
- [ ] SNAILS sqlite database collection: Google Drive link in `SNAILS_Artifacts/databases/snails_sqlite_download_link.txt`; size TBD
- [x] Spider databases and BIRD dev databases: already downloaded locally

**Validation**: run every gold SQL on its sqlite database. Log every example that errors, times out or returns an empty result; never drop them silently. SNAILS' sqlite SQL was translated from T-SQL, so check it closely.

## 6. Timeline and phases

Target completion: **2026-11-24**, nine days later than the original plan, in exchange for multi-turn RL. Two key checkpoints: the SNAILS floor check on 11/4, and the agent running end-to-end on 11/9. Every phase produces results that stand on their own, so even if multi-turn RL is not working, the earlier results are still usable.

| Phase | Dates | Gate |
| --- | --- | --- |
| 0 Environment and data | 10/23–10/25 | — |
| 1 Data pipeline and evaluation | 10/25–10/30 | gold-as-prediction scores 1.0 |
| 2 Single-turn baseline + SFT | 10/30–11/4 | **11/4 SNAILS floor check** |
| 3 Tool environment + cold-start | 11/4–11/9 | **11/9 agent runs end-to-end** |
| 4 Multi-turn GRPO | 11/9–11/19 | Core comparison table filled |
| 5 Analysis and packaging | 11/19–11/24 | — |

**Division of work**: the user writes the core code; Claude explains the principles, provides skeletons and hints, and reviews. Claude may write boilerplate such as download scripts and configs, but must explain it afterward. All code comments in English; commit in small steps; every experiment has a config.

### Phase 0: Environment and data (10/23–10/25)

- [x] Local conda env; download BIRD dev, BIRD train metadata, Spider, SNAILS
- [x] Qwen3-0.6B zero-shot smoke test
- [ ] Sign up for RunPod and add credit (~$10 to start)
- [ ] Download BIRD train databases and SNAILS sqlite databases on the GPU machine
- [ ] First commit (made by the user)

### Phase 1: Data pipeline and evaluation (10/25–10/30)

- [ ] `parse_sql.py`: SQL → schema links, handling aliases, subqueries and CTEs, unqualified columns, wildcards, quoted identifiers (**user writes**)
- [ ] `eval.py`: EX (result-set comparison with timeout) + linking F1, with per-example output (**user writes**)
- [ ] Unified jsonl; train/val split by database; verify every gold SQL executes
- [ ] Data statistics: example counts, table and column counts, prompt length distribution

Checkpoint: running `eval.py` with gold as the prediction gives EX = 1.0 and F1 = 1.0; parse success rate >98%; manually check 30 samples.

### Phase 2: Single-turn baseline and SFT (10/30–11/4)

- [ ] Zero-shot: 0.6B and 1.7B, on both BIRD dev and SNAILS
- [ ] Frontier API few-shot: 200 sampled examples; record score, latency and cost per 1k calls
- [ ] Overfit test: train 0.6B on 100 examples; training accuracy should approach 100%
- [ ] Single-turn LoRA SFT: 1.7B main config (rank 16, alpha 32, all linear layers, lr 1e-4, cosine schedule, 2 epochs) + ablations: 0.6B, and S1 vs S2 serialization

**Checkpoint 11/4: SNAILS floor check.** If single-turn SFT scores below ~10% EX on SNAILS, switch to the fallback (see Risks).

### Phase 3: Tool environment and cold-start (11/4–11/9)

- [ ] Read SkyRL-SQL's environment and agent loop code, and TRUST-SQL's Phase-Aware GRPO (Claude prepares a question list first)
- [ ] Tool environment class + unit tests: read-only, timeouts, output truncation (**user writes**)
- [ ] Reward function + unit tests (**user writes**)
- [ ] Generate cold-start trajectories by script; LoRA SFT on them; check the loss mask token by token
- [ ] Multi-turn evaluation loop (vLLM + environment)

Checkpoint 11/9: the cold-start agent runs end-to-end on BIRD dev, with numbers to compare against single-turn SFT.

### Phase 4: GRPO (11/9–11/19)

- [ ] Run verl's official GSM8K + GRPO example (0.6B, a few dozen steps) to confirm the environment, rollout and logging work
- [ ] Single-turn GRPO (EX reward), to get familiar with verl configs and training curves
- [ ] Filter the GRPO pool: sample 8 times per example, keep examples with 0 < correct < 8
- [ ] Multi-turn GRPO: variant A (EX only) vs B (EX + 0.3·linking), 2 seeds each

Checkpoint: core comparison table filled, with W&B curves attached.

### Phase 5: Analysis and packaging (11/19–11/24)

- [ ] Error analysis: sample 50 errors and categorize by hand (missing table, same-named column on the wrong table, insufficient exploration, tool-call errors, bad gold)
- [ ] Behavior analysis: tool calls vs. schema size and naturalness level, before and after RL
- [ ] README, W&B report, blog post, headline summary

**If 11/15 is a hard deadline**: drop the frontier API baseline and the S1 vs S2 ablation, run multi-turn GRPO with 1 seed, and postpone the blog post.

## 7. Experiment matrix and evaluation protocol

Every conclusion comes from a paired comparison: same questions, greedy decoding, reported with 95% confidence intervals. The core table to fill in:

| Model | BIRD dev EX | BIRD dev link F1 | SNAILS Native EX | SNAILS Least EX | Avg input tokens (SNAILS) | Avg tool calls |
| --- | --- | --- | --- | --- | --- | --- |
| Zero-shot 1.7B (single-turn) |  |  |  |  |  | 0 |
| Single-turn LoRA SFT 1.7B |  |  |  |  |  | 0 |
| Agent: cold-start SFT |  |  |  |  |  |  |
| Agent + GRPO (A: EX only) |  |  |  |  |  |  |
| Agent + GRPO (B: EX + 0.3·linking) |  |  |  |  |  |  |
| Frontier API few-shot (200 examples) |  |  |  |  |  | 0 |

**Single-turn SFT ablations** (report BIRD dev and SNAILS Native only): 0.6B vs 1.7B; S1 vs S2 serialization.

### Statistical protocol

- Paired bootstrap: resample questions with replacement 10,000 times to get a 95% CI on the score difference between two models; also run McNemar's test for 0/1 metrics such as EX.
- The core GRPO comparison (A vs B) uses at least 2 seeds per variant; report the mean and each seed's number.
- Report SNAILS results only grouped by naturalness level and domain, never per db_id (each SBODemoUS module has only 10 questions).
- Native and Least are the same 503 questions, so the "naming causes the drop" conclusion is paired too.

**Curves to watch during training**: reward mean and variance, val EX, average turns and output length, average tables and columns referenced (hacking signal), KL and entropy (collapse), tool-call error rate.

## 8. Risks and mitigations

The two biggest risks are a SNAILS floor effect and the engineering of multi-turn verl training; the 11/4 and 11/9 checkpoints surface them early.

| Risk | Signal | Mitigation |
| --- | --- | --- |
| SNAILS floor effect | Single-turn SFT scores < 10% EX on SNAILS at 11/4 | Use link F1 as the main OOD metric; switch to 4B; or use the renamed Spider dev; report the finding honestly |
| verl multi-turn integration is hard | Still not running 3 days after 11/9 | Reuse SkyRL-SQL's agent loop directly; if that fails, ship the single-turn GRPO results and push multi-turn back |
| Loss mask bug | Training loss looks normal but the model starts fabricating tool output | Unit test that prints the mask token by token; rerun on every template change |
| Reward hacking | Average referenced columns rise, turns pinned at the limit | Precision is in the reward; 8-turn cap; add a small tool-call penalty if needed |
| GRPO shows no gain | Reward curve flat, test scores unchanged | First check that data filtering worked and within-group variance is not 0; this is itself a finding for the blog |
| 24GB OOM | Crash during rollout or training | Use the A40 48GB for GRPO; shorten tool-output truncation; smaller group size; start with 0.6B |
| SNAILS sqlite gold SQL is broken | Errors or empty results during validation | Log separately; state in the report which examples were excluded; never drop silently |
| RunPod cost overrun | Bill exceeds the ~$50 budget | Terminate when done (a stopped pod still bills for storage); keep data and checkpoints on a network volume; do non-GPU work on the Mac; training scripts support resuming from checkpoints |
| Schedule overrun | Cold-start not done by 11/9 | Cut scope using the reduced plan at the end of the Timeline section |

## 9. Learning goals and next steps

By the end of the project, you should be able to explain each of the following against your own code. They are also among the most common topics in agentic RL interviews.

- [ ] How LoRA works, and how to choose rank, alpha and target layers
- [ ] How to implement completion-only loss and the loss mask for multi-turn conversations
- [ ] GRPO vs. PPO: group-relative advantage, no value model, the role of the KL term
- [ ] Why all-correct and all-wrong groups have no gradient (the basis for data filtering)
- [ ] The structure of a multi-turn rollout: generate → parse tool call → execute → append tool output → continue generating
- [ ] Sparse vs. dense rewards, and credit assignment across turns
- [ ] Signals of reward hacking and how to respond
- [ ] Why cold-start SFT is needed, and its trade-off with RL
- [ ] Paired tests and multiple seeds: telling real gains from noise

### Next steps

1. Answer the three comprehension-check questions and do the sqlglot warm-up (parse a JOIN with aliases; print the tables, aliases and columns).
2. Sign up for RunPod and add ~$10. Before phase 2, rent an A40, run `nvidia-smi` and install vLLM to validate the environment (confirm the price before creating; Terminate when done).
3. Start `parse_sql.py`: write the unit tests first, then the implementation.

## Sources

- [SkyRL-SQL](https://novasky-ai.notion.site/skyrl-sql) · [Train SQL Data Agent with Multi-Turn RL (NeurIPS 2025)](https://neurips.cc/virtual/2025/128018)
- [TRUST-SQL](https://arxiv.org/abs/2603.16448)
- [SQL-Trail (ACL 2026)](https://aclanthology.org/2026.acl-long.1677/)
- [AutoLink (AAAI 2026)](https://arxiv.org/abs/2511.17190)
- [Graph-Reward-SQL](https://aclanthology.org/2025.findings-emnlp.694.pdf) · [PaVeRL-SQL](https://arxiv.org/pdf/2509.07159)
- [SNAILS](https://rmarcus.info/dbscholar/papers/7069) · [SNAILS artifacts](https://github.com/KyleLuoma/SNAILS)
- [RunPod pricing](https://docs.runpod.io/pods/pricing) (GPU prices checked 2026-10-24)

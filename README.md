# SQL Data Agent

A multi-turn SQL agent (Qwen3-1.7B) that explores SQLite databases through tools instead of
reading the full schema, trained with LoRA SFT cold-start followed by multi-turn GRPO (verl).

The project compares an execution-only reward with execution + dense schema-linking reward, and
evaluates in-distribution on BIRD dev and out-of-distribution on SNAILS (enterprise schemas with
obscure identifiers).

Work in progress; see [plan.md](plan.md).

## Setup

```bash
conda create -n schemalink python=3.11 && conda activate schemalink
pip install -r requirements.txt && pip install -e .
bash scripts/download_data.sh
pytest -q
```

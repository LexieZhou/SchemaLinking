#!/usr/bin/env bash
# Download all raw data for phase 1 into data/raw/ (git-ignored).
# Usage (from repo root, inside the schemalink env):
#   bash scripts/download_data.sh          # BIRD train metadata only (~1MB instead of 8.9GB)
#   FULL_BIRD=1 bash scripts/download_data.sh   # also fetch BIRD train sqlite DBs (needed for S3)
set -euo pipefail

RAW=data/raw
mkdir -p "$RAW"/{bird,spider,snails}

# --- BIRD dev (346MB incl. dev_databases.zip, 2024-06-27 revision) ---
if [ ! -d "$RAW/bird/dev_20240627" ]; then
  curl -L -o "$RAW/bird/dev.zip" https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip
  unzip -q "$RAW/bird/dev.zip" -d "$RAW/bird" -x '__MACOSX/*'
  unzip -q "$RAW/bird/dev_20240627/dev_databases.zip" -d "$RAW/bird/dev_20240627" -x '__MACOSX/*'
fi

# --- BIRD train ---
if [ "${FULL_BIRD:-0}" = "1" ]; then
  # Full archive: ~9GB download, 30GB+ once the databases are unzipped
  if [ ! -d "$RAW/bird/train/train_databases" ]; then
    curl -L -o "$RAW/bird/train.zip" https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip
    unzip -q "$RAW/bird/train.zip" -d "$RAW/bird" -x '__MACOSX/*'
    unzip -q "$RAW/bird/train/train_databases.zip" -d "$RAW/bird/train" -x '__MACOSX/*'
  fi
elif [ ! -f "$RAW/bird/train/train.json" ]; then
  # Metadata only, via HTTP range requests
  python -I scripts/fetch_bird_train_meta.py --out "$RAW/bird"
fi

# --- Spider 1.0 (official Google Drive release, incl. tables.json and sqlite DBs) ---
if [ ! -d "$RAW/spider/spider_data" ]; then
  gdown 1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J -O "$RAW/spider/spider_data.zip"
  # Expected SHA256: 00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b
  unzip -q "$RAW/spider/spider_data.zip" -d "$RAW/spider" -x '__MACOSX/*'
fi

# --- SNAILS artifacts (NL-SQL pairs, naturalness labels, rename crosswalks, Spider-format schemas) ---
if [ ! -d "$RAW/snails/SNAILS" ]; then
  git clone --depth 1 https://github.com/KyleLuoma/SNAILS "$RAW/snails/SNAILS"
fi
if [ ! -d "$RAW/snails/nl_sql" ]; then
  unzip -q "$RAW/snails/SNAILS/SNAILS_Artifacts/nl_sql_question_query_pairs/SNAILS NL-SQL Pairs.zip" \
    -d "$RAW/snails/nl_sql"
fi

echo "done:"; du -sh "$RAW"/*

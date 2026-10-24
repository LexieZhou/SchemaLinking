"""Extract only the metadata files from BIRD train.zip (8.9GB), skipping the sqlite databases.

Uses HTTP range requests (remotezip), so only the bytes of the requested members are transferred.
The databases (train_databases.zip) are only needed for the S3 ablation (sample values) in phase 3.
"""
import argparse
from pathlib import Path

from remotezip import RemoteZip

URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip"
MEMBERS = ["train/train.json", "train/train_tables.json", "train/train_gold.sql"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/bird")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with RemoteZip(URL) as z:
        for m in MEMBERS:
            z.extract(m, out)
            print("extracted", out / m)


if __name__ == "__main__":
    main()

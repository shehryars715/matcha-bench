"""Fingerprint the user-level split, so anyone with MATCHA access can check they rebuilt the same one.

    python scripts/split_fingerprint.py                 # print users and a SHA-256 per split
    python scripts/split_fingerprint.py --write-ids     # also write the ID lists to outputs/splits/

The fingerprint of a split is the SHA-256 of its sorted user IDs, one per line (UTF-8, "\\n" after each).
The README lists the fingerprints of the split used for every result. The ID lists themselves are Weibo
user IDs of people in a mental-health dataset: they stay under the MATCHA data use agreement and are
never committed (outputs/ is git-ignored); share them only with people who have MATCHA access.
"""

import argparse
import hashlib

import pandas as pd

import _bootstrap  # noqa: F401
from matcha_bench.config import load_config, resolve_path


def fingerprint(ids) -> str:
    return hashlib.sha256("".join(f"{i}\n" for i in sorted(map(str, ids))).encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--write-ids", action="store_true", help="write <split>_user_ids.txt to outputs/splits/")
    args = ap.parse_args()
    cfg = load_config(resolve_path(args.config))
    users = pd.read_parquet(resolve_path(cfg["processed_dir"]) / "users.parquet", columns=["user_id", "split"])
    out = resolve_path("outputs/splits")
    for split in ("train", "dev", "test"):
        ids = sorted(users.loc[users["split"] == split, "user_id"].astype(str))
        print(f"{split:>5}: {len(ids):>6} users  sha256 {fingerprint(ids)}")
        if args.write_ids:
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{split}_user_ids.txt").write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
    if args.write_ids:
        print(f"ID lists in {out} (DUA-restricted: do not publish)")


if __name__ == "__main__":
    main()

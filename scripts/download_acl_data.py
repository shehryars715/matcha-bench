"""Download the public downstream datasets used by Chinese MentalBERT into data/raw/acl/."""

import argparse
from pathlib import Path

import requests

import _bootstrap  # noqa: F401
from matcha_bench.config import resolve_path
from matcha_bench.tasks.acl_tasks import RAW_URLS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/raw/acl")
    args = ap.parse_args()
    out = resolve_path(args.out)
    for rel, url in RAW_URLS.items():
        dest = Path(out) / rel
        if dest.exists():
            print(f"exists  {rel}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
        print(f"fetched {rel} ({len(r.content) / 1e6:.1f} MB)")
    print(f"C2D2 is shared on request by its authors; put train/dev/test.csv (text,label) in {Path(out) / 'c2d2'}")


if __name__ == "__main__":
    main()

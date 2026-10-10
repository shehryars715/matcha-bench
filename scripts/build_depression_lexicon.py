"""Step 3 (faithful lexicon): download and convert the Chinese depression-domain lexicon
(Li et al., 2020) with the same filter as the Chinese MentalBERT code
(抑郁倾向 == 1, Chinese characters only)."""

import argparse
import tempfile
from pathlib import Path

import requests

import _bootstrap  # noqa: F401
from matcha_bench.config import load_config, resolve_path
from matcha_bench.lexicon import save_lexicon
from matcha_bench.lexicon.io import DEPRESSION_LEXICON_URL, convert_depression_lexicon


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/lexicon.yaml")
    args = ap.parse_args()
    cfg = load_config(resolve_path(args.config))
    r = requests.get(DEPRESSION_LEXICON_URL, timeout=120)
    r.raise_for_status()
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "lexicon.csv"
        csv.write_bytes(r.content)
        words = convert_depression_lexicon(csv)
    out = resolve_path(cfg["depression"]["output"])
    save_lexicon(words, out, header=f"Chinese depression-domain lexicon (Li et al., 2020)\nsource: {DEPRESSION_LEXICON_URL}")
    print(f"wrote {len(words)} words to {out}")


if __name__ == "__main__":
    main()

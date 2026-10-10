"""Step 3b (extension lexicon): turn word-level filtering decisions into the multi-disorder
lexicon used for masking.

The annotation file is a CSV with columns ``word,category`` that lists every candidate judged to
be mental-health related. Categories: emotion, cognition, somatic, behavior, clinical,
community, social. It can come from an LLM annotator, a human annotator, or
``filter_lexicon_deepseek.py`` (rows with keep == 1).
"""

import argparse

import pandas as pd

import _bootstrap  # noqa: F401
from matcha_bench.config import load_config, resolve_path
from matcha_bench.lexicon import save_lexicon
from matcha_bench.lexicon.io import apply_word_annotations


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/lexicon.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(resolve_path(args.config), args.set)
    md = cfg["multidisorder"]

    candidates = pd.read_csv(resolve_path(cfg["mining"]["output"]))
    annotations = pd.read_csv(resolve_path(md["annotations"]))
    annotated, unmatched = apply_word_annotations(candidates, annotations)
    if unmatched:
        print(f"warning: {len(unmatched)} annotated words are not candidates: {' '.join(unmatched[:30])}")

    kept = annotated[(annotated["keep"] == 1) & ~annotated["category"].isin(md.get("exclude_categories") or [])]
    annotated.to_csv(resolve_path(md["annotated_candidates"]), index=False, encoding="utf-8")
    save_lexicon(kept["word"].unique(), resolve_path(md["output"]), header=md.get("header"))

    print(kept.groupby(["disorder", "category"]).size().unstack(fill_value=0).to_string())
    print(f"kept {kept['word'].nunique()} of {candidates['word'].nunique()} unique candidate words "
          f"-> {resolve_path(md['output'])}")


if __name__ == "__main__":
    main()

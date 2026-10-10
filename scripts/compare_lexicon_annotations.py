"""Agreement (Cohen's kappa) between the LLM word annotations and the DeepSeek votes.

    python scripts/filter_lexicon_deepseek.py          # second annotator
    python scripts/compare_lexicon_annotations.py
"""

import argparse

import pandas as pd

import _bootstrap  # noqa: F401
from matcha_bench.config import load_config, resolve_path
from matcha_bench.lexicon.agreement import annotation_agreement
from matcha_bench.utils import save_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/lexicon.yaml")
    ap.add_argument("--out", default="lexicons/annotation_agreement")
    args = ap.parse_args()
    cfg = load_config(resolve_path(args.config))

    candidates = pd.read_csv(resolve_path(cfg["mining"]["output"]))
    ann_a = pd.read_csv(resolve_path(cfg["multidisorder"]["annotations"]))
    ann_b = pd.read_csv(resolve_path(cfg["deepseek"]["annotations"]))
    report, disagreements = annotation_agreement(candidates, ann_a, ann_b)

    out = resolve_path(args.out)
    save_json(report, out.with_suffix(".json"))
    disagreements.to_csv(out.with_name(out.name + "_disagreements.csv"), index=False, encoding="utf-8")
    for level in ("word_level", "pair_level"):
        r = report[level]
        print(f"{level:>10}: n={r['n']}  agreement={r['percent_agreement']:.3f}  kappa={r['cohen_kappa']:.3f}  "
              f"both_keep={r['both_keep']}  only_A={r['only_a_keeps']}  only_B={r['only_b_keeps']}")
    print(f"category agreement on shared keeps: {report['category_agreement_on_shared_keeps']}")
    print(f"{len(disagreements)} disagreements -> {out.name}_disagreements.csv")


if __name__ == "__main__":
    main()

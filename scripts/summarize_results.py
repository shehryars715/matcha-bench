"""Step 7: results tables (mean ± std over seeds) and paired-bootstrap significance tests.

    python scripts/summarize_results.py
    python scripts/summarize_results.py --compare matcha-deepseek matcha-random
"""

import argparse

import pandas as pd

import _bootstrap  # noqa: F401
from matcha_bench.config import load_config, resolve_path
from matcha_bench.stats import collect_results, compare_models, summarise


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/finetune.yaml")
    ap.add_argument("--metric", default="f1", choices=["f1", "macro_f1", "accuracy"])
    ap.add_argument("--compare", nargs=2, metavar=("MODEL_A", "MODEL_B"))
    ap.add_argument("--out", default="outputs/results")
    args = ap.parse_args()
    cfg = load_config(resolve_path(args.config))
    root, out = resolve_path(cfg["output_root"]), resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = collect_results(root)
    if results.empty:
        print(f"No results under {root}")
        return
    results.to_csv(out / "all_runs.csv", index=False)
    table = summarise(results, args.metric)
    table.to_csv(out / f"summary_{args.metric}.csv")
    print(table.to_string())

    if args.compare:
        a, b = args.compare
        frames = [compare_models(root, t, a, b) for t in sorted(results["task"].unique())]
        frames = [f for f in frames if not f.empty]
        if frames:
            sig = pd.concat(frames, ignore_index=True)
            sig.to_csv(out / f"significance_{a}_vs_{b}.csv", index=False)
            print(sig.groupby("task")[["delta", "p_value"]].mean().to_string())


if __name__ == "__main__":
    main()

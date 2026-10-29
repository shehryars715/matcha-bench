"""Class-prior correction of every baseline, tuned on dev (configs/prior_correction.yaml).

    python scripts/prior_correction.py

For each run with saved dev and test scores: tau is picked on dev by macro-F1 and the corrected test
F1 is reported next to the raw one (matcha_bench.prior). Runs not downloaded yet are listed as
missing. Writes outputs/results/prior/{runs.csv,summary.md}.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from matcha_bench.config import REPO_ROOT, load_config, resolve_path
from matcha_bench.prior import log_shift, macro_f1, tune_tau
from matcha_bench.results import expand_roots, model_seeds, run_path


def load_scores(path: Path, key: str):
    z = np.load(path)
    order = np.argsort(z["indices"], kind="stable") if "indices" in z.files else np.arange(len(z["labels"]))
    return z[key][order].astype(np.float64), z["labels"][order].astype(int)


def find(roots, rel: str, files: dict) -> Path | None:
    for root in roots:  # copies restored into later sessions are identical; the first one is used
        if all((root / rel / files[s]).exists() for s in ("dev", "test")):
            return root / rel
    return None


def reported_scores(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {(r["task"], r["model"], int(float(r["seed"]))): float(r["f1"]) for r in csv.DictReader(f)
                if r["table"] == "main"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/prior_correction.yaml")
    args = ap.parse_args()
    cfg = load_config(resolve_path(args.config))
    roots = expand_roots(cfg["roots"], REPO_ROOT)
    g = cfg["tau_grid"]
    grid = np.round(np.arange(g["start"], g["stop"] + 1e-9, g["step"]), 4)
    reported = reported_scores(resolve_path(cfg["reported_runs"]))
    rows, missing = [], []
    for task in cfg["tasks"]:
        meta = json.loads((resolve_path("data/tasks") / task / "meta.json").read_text(encoding="utf-8"))
        names, k = meta["label_names"], len(meta["label_names"])
        train_mix = np.array([meta["label_distribution"]["train"][n] for n in names], dtype=float)
        train_mix /= train_mix.sum()
        for key, model in cfg["models"].items():
            for seed in model_seeds(model, task, cfg["seeds"]):
                rel = run_path(key, model, task, seed)
                run_dir = find(roots, rel, model["files"])
                if run_dir is None:
                    missing.append(key)
                    continue
                dev, dev_y = load_scores(run_dir / model["files"]["dev"], model["files"]["key"])
                test, test_y = load_scores(run_dir / model["files"]["test"], model["files"]["key"])
                shift = log_shift(dev_y, k, None if model["trained_toward"] == "uniform" else train_mix)
                tau, dev_corr = tune_tau(dev, dev_y, shift, grid, meta["average"])
                rows.append({
                    "task": task, "model": key, "seed": seed, "tau": tau,
                    "dev_raw": 100 * macro_f1(dev_y, dev, meta["average"]), "dev_corrected": 100 * dev_corr,
                    "test_raw": 100 * macro_f1(test_y, test, meta["average"]),
                    "test_corrected": 100 * macro_f1(test_y, test + tau * shift, meta["average"]),
                    "test_reported": reported.get((task, key, seed)), "source": str(run_dir.relative_to(REPO_ROOT)),
                })
    out = resolve_path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out / "runs.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["task"], r["model"])].append(r)
    lines = ["| task | model | runs | test raw | test corrected | tau (mean) | reported raw |", "|---|---|---|---|---|---|---|"]
    fmt = lambda v: f"{np.mean(v):.2f} ± {np.std(v, ddof=1):.2f}" if len(v) > 1 else f"{v[0]:.2f}"
    for (task, key), rs in groups.items():
        rep = [r["test_reported"] for r in rs if r["test_reported"] is not None]
        lines.append(f"| {task} | {cfg['models'][key]['name']} | {len(rs)} | {fmt([r['test_raw'] for r in rs])} | "
                     f"{fmt([r['test_corrected'] for r in rs])} | {np.mean([r['tau'] for r in rs]):.2f} | "
                     f"{fmt(rep) if rep else '—'} |")
    if missing:
        counts = ", ".join(f"{m} {missing.count(m)}" for m in dict.fromkeys(missing))
        lines += ["", f"Runs without dev and test scores yet ({len(missing)}): {counts}"]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

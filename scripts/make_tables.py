"""Results tables for the paper, built from finished runs (which runs: configs/results.yaml).

    python scripts/make_tables.py                 # all tables + 10,000-resample paired bootstrap
    python scripts/make_tables.py --resamples 1000

Writes to outputs/results/paper/:
  runs.csv              every run used (task, model, seed, F1, where it came from, checked against predictions)
  main.{md,tex,csv}     test F1 on the MATCHA tasks, mean ± SD over seeds
  zeroshot.md           zero-shot rows: share of answers that name a label, F1 on those only
  bootstrap.{csv,md}    every model pair per MATCHA task: difference in mean F1, 95% CI, p (raw and Holm)
  sanity_<name>.{md,tex}, sanity_gaps.csv   each reproduction vs the numbers its paper reports
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from matcha_bench.config import REPO_ROOT, load_config, resolve_path
from matcha_bench.finetune.metrics import classification_metrics
from matcha_bench.results import (bootstrap_macro_f1, check_run, compare, expand_roots, find_run, holm,
                                  model_seeds)


def collect(cfg, roots, table: str, tasks, model_keys, need_predictions: bool) -> list:
    runs = []
    for task, key in itertools.product(tasks, model_keys):
        model = cfg["models"][key]
        for seed in model_seeds(model, task, cfg["seeds"]):
            run = find_run(roots, key, model, task, seed)
            if run is None:
                print(f"  missing: {table} {task} {key} seed {seed}")
                continue
            if need_predictions and not run.has_predictions:
                raise SystemExit(f"{run.path} has no {run.path / 'test_predictions.npz'}: download it first "
                                 "(python -m kaggle kernels output ... --file-pattern test_predictions)")
            run.checked = check_run(run) is not None
            run.table = table
            runs.append(run)
    return runs


def cell(values: list, expected: int) -> str:
    if not values:
        return ""
    v = 100 * np.asarray(values)
    text = f"{v.mean():.2f} ± {v.std(ddof=1):.2f}" if len(v) > 1 else f"{v[0]:.2f}"
    return text + (f" (n={len(v)})" if len(v) < expected else "")


def score_table(cfg, runs, tasks, model_keys) -> pd.DataFrame:
    rows = []
    for key in model_keys:
        model = cfg["models"][key]
        row = {"group": model["group"], "model": model["name"]}
        for task in tasks:
            vals = [r.f1 for r in runs if r.model == key and r.task == task]
            row[cfg["task_names"][task]] = cell(vals, len(model_seeds(model, task, cfg["seeds"])))
        rows.append(row)
    return pd.DataFrame(rows)


def best_mask(cfg, runs, tasks, model_keys, skip_groups=("Zero-shot", "Reference")) -> dict:
    """(model, task) cells with the highest mean among the trained baselines, for bolding."""
    best = {}
    for task in tasks:
        means = {k: np.mean([r.f1 for r in runs if r.model == k and r.task == task]) for k in model_keys
                 if cfg["models"][k]["group"] not in skip_groups and any(r.model == k and r.task == task for r in runs)}
        if means:
            best[(max(means, key=means.get), task)] = True
    return best


def to_markdown(df: pd.DataFrame, bold: set | None = None) -> str:
    bold = bold or set()
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    lines = []
    for i, row in df.iterrows():
        cells = [f"**{v}**" if (i, c) in bold and v else str(v) for c, v in row.items()]
        lines.append("| " + " | ".join(cells) + " |")
    return head + "\n".join(lines) + "\n"


def to_latex(df: pd.DataFrame, caption: str, label: str, bold: set | None = None) -> str:
    bold = bold or set()
    esc = lambda s: str(s).replace("_", r"\_").replace("&", r"\&").replace("±", r"$\pm$").replace("%", r"\%")
    cols = [c for c in df.columns if c != "group"]
    lines = [r"\begin{table*}[t]", r"\centering", r"\small",
             r"\begin{tabular}{l" + "c" * (len(cols) - 1) + "}", r"\toprule",
             " & ".join(esc(c) for c in cols) + r" \\", r"\midrule"]
    group = None
    for i, row in df.iterrows():
        if "group" in df.columns and row["group"] != group:
            if group is not None:
                lines.append(r"\midrule")
            group = row["group"]
            lines.append(rf"\multicolumn{{{len(cols)}}}{{l}}{{\textit{{{esc(group)}}}}} \\")
        cells = [rf"\textbf{{{esc(row[c])}}}" if (i, c) in bold and row[c] else esc(row[c]) for c in cols]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{esc(caption)}}}", rf"\label{{{label}}}",
              r"\end{table*}"]
    return "\n".join(lines) + "\n"


def bold_cells(df, cfg, runs, tasks, keys) -> set:
    best = best_mask(cfg, runs, tasks, keys)
    return {(i, cfg["task_names"][t]) for i, k in enumerate(keys) for t in tasks if (k, t) in best}


def zeroshot_table(cfg, runs, tasks) -> pd.DataFrame:
    rows = []
    for key in sorted({r.model for r in runs if cfg["models"][r.model]["group"] == "Zero-shot"},
                      key=cfg["main"]["models"].index):
        row = {"model": cfg["models"][key]["name"]}
        for task in tasks:
            run = next((r for r in runs if r.model == key and r.task == task), None)
            if run is None:
                continue
            labels, preds = run.predictions()
            answered = preds >= 0
            names = run.metrics["label_names"]
            f1_answered = classification_metrics(labels[answered], preds[answered], run.metrics["problem_type"],
                                                 run.test["average"], names, list(range(len(names))))["f1"]
            t = cfg["task_names"][task]
            row[f"{t} F1"] = f"{100 * run.f1:.2f}"
            row[f"{t} answered %"] = f"{100 * answered.mean():.1f}"
            row[f"{t} F1 (answered)"] = f"{100 * f1_answered:.2f}"
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_all(cfg, runs, tasks, keys, n_resamples, seed) -> pd.DataFrame:
    rows = []
    for task in tasks:
        task_runs = [r for r in runs if r.task == task]
        loaded = [r.predictions() for r in task_runs]
        labels = loaded[0][0]
        for r, (lab, _) in zip(task_runs, loaded):
            if not np.array_equal(lab, labels):
                raise SystemExit(f"{r.path}: test labels differ from the other runs of {task}")
        n_classes = len(task_runs[0].metrics["label_names"])
        observed, boot = bootstrap_macro_f1(labels, [p for _, p in loaded], n_classes, n_resamples, seed)
        for r, o in zip(task_runs, observed):
            if abs(o - r.f1) > 1e-9:
                raise SystemExit(f"{r.path}: bootstrap F1 {o:.6f} != metrics.json {r.f1:.6f}")
        cols = {k: [i for i, r in enumerate(task_runs) if r.model == k] for k in keys}
        task_rows = []
        for a, b in itertools.combinations([k for k in keys if cols[k]], 2):
            ia, ib = cols[a], cols[b]
            res = compare(observed[ia], boot[:, ia], observed[ib], boot[:, ib])
            task_rows.append({"task": cfg["task_names"][task], "a": a, "b": b, "pairing": "pooled",
                              "n_a": len(ia), "n_b": len(ib), **res})
            if len(ia) == len(ib) > 1:  # seed pairs by position (s45 stands in for s44 where it replaced it)
                for pos, (x, y) in enumerate(zip(ia, ib)):
                    res = compare(observed[[x]], boot[:, [x]], observed[[y]], boot[:, [y]])
                    task_rows.append({"task": cfg["task_names"][task], "a": a, "b": b,
                                      "pairing": f"s{task_runs[x].seed}-s{task_runs[y].seed}", "n_a": 1, "n_b": 1,
                                      **res})
        pooled = [r for r in task_rows if r["pairing"] == "pooled"]
        for r, p in zip(pooled, holm([r["p_value"] for r in pooled])):
            r["p_holm"] = p
        rows += task_rows
    df = pd.DataFrame(rows)
    for c in ("delta", "ci_low", "ci_high"):
        df[c] = 100 * df[c]
    return df


def bootstrap_markdown(df: pd.DataFrame, cfg, alpha: float) -> str:
    names = {k: m["name"] for k, m in cfg["models"].items()}
    out = []
    for task, g in df[df["pairing"] == "pooled"].groupby("task", sort=False):
        out.append(f"### {task}\n\n| A | B | Δ F1 (A − B) | 95% CI | p | p (Holm) | per-seed pairs with p < {alpha} |\n"
                   "|---|---|---|---|---|---|---|")
        seeds = df[(df["task"] == task) & (df["pairing"] != "pooled")]
        for _, r in g.iterrows():
            s = seeds[(seeds["a"] == r["a"]) & (seeds["b"] == r["b"])]
            per_seed = f"{(s['p_value'] < alpha).sum()}/{len(s)}" if len(s) else "–"
            out.append(f"| {names[r['a']]} | {names[r['b']]} | {r['delta']:+.2f} | [{r['ci_low']:+.2f}, "
                       f"{r['ci_high']:+.2f}] | {r['p_value']:.4f} | {r['p_holm']:.4f} | {per_seed} |")
        out.append("")
    return "\n".join(out)


def sanity_tables(cfg, roots, out: Path, all_runs: list) -> None:
    gaps = []
    for spec in cfg.get("sanity", []):
        runs = collect(cfg, roots, f"sanity_{spec['name']}", spec["tasks"], spec["models"], need_predictions=False)
        all_runs += runs
        df = score_table(cfg, runs, spec["tasks"], spec["models"])
        for rep in spec.get("reported", []):
            df.loc[len(df)] = {"group": "Reported", "model": rep["name"],
                               **{cfg["task_names"][t]: f"{rep['values'][t]:.2f}" if t in rep["values"] else ""
                                  for t in spec["tasks"]}}
            for task, value in rep["values"].items():
                vals = [r.f1 for r in runs if r.model == rep["ours"] and r.task == task]
                if vals:
                    gaps.append({"table": spec["name"], "task": cfg["task_names"][task], "ours": rep["ours"],
                                 "ours_f1": 100 * np.mean(vals), "n_seeds": len(vals), "reported": rep["name"],
                                 "reported_f1": value, "gap": 100 * np.mean(vals) - value, "source": rep["source"]})
        (out / f"sanity_{spec['name']}.md").write_text(f"## {spec['title']}\n\n" + to_markdown(df), encoding="utf-8")
        (out / f"sanity_{spec['name']}.tex").write_text(
            to_latex(df, spec["title"], f"tab:sanity-{spec['name'].replace('_', '-')}"), encoding="utf-8")
        print(f"\n{spec['title']}\n" + df.drop(columns="group").to_string(index=False))
    pd.DataFrame(gaps).round(2).to_csv(out / "sanity_gaps.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/results.yaml")
    ap.add_argument("--resamples", type=int, help="bootstrap resamples instead of the config's")
    args = ap.parse_args()
    cfg = load_config(resolve_path(args.config))
    roots = expand_roots(cfg["roots"], REPO_ROOT)
    out = resolve_path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    boot_cfg = cfg["bootstrap"]
    n_resamples = args.resamples or boot_cfg["n_resamples"]
    print(f"{len(roots)} output roots")

    tasks, keys = cfg["main"]["tasks"], cfg["main"]["models"]
    runs = collect(cfg, roots, "main", tasks, keys, need_predictions=True)
    main_df = score_table(cfg, runs, tasks, keys)
    bold = bold_cells(main_df, cfg, runs, tasks, keys)
    main_df.to_csv(out / "main.csv", index=False)
    (out / "main.md").write_text(to_markdown(main_df.drop(columns="group"), bold), encoding="utf-8")
    caption = ("Test macro-F1 (%) on the MATCHA tasks, mean ± SD over three training seeds (one run for TF-IDF "
               "and the zero-shot rows). Bold: best trained baseline per task.")
    (out / "main.tex").write_text(to_latex(main_df, caption, "tab:main", bold), encoding="utf-8")
    print("\n" + main_df.drop(columns="group").to_string(index=False))

    zs = zeroshot_table(cfg, runs, tasks)
    (out / "zeroshot.md").write_text(to_markdown(zs), encoding="utf-8")
    print("\n" + zs.to_string(index=False))

    boot = bootstrap_all(cfg, runs, tasks, keys, n_resamples, boot_cfg["seed"])
    boot.round(6).to_csv(out / "bootstrap.csv", index=False)
    (out / "bootstrap.md").write_text(
        f"Paired bootstrap, {n_resamples:,} resamples of each test set (the same resamples for every model). "
        "Δ = difference in mean test macro-F1 over each model's seeds; p two-sided; Holm over all pairs of the "
        "task.\n\n" + bootstrap_markdown(boot, cfg, boot_cfg["alpha"]), encoding="utf-8")

    all_runs = list(runs)
    sanity_tables(cfg, roots, out, all_runs)
    pd.DataFrame([{"table": r.table, "task": r.task, "model": r.model, "seed": r.seed, "f1": round(100 * r.f1, 4),
                   "checked_against_predictions": r.checked, "copies": r.copies,
                   "source": r.path.relative_to(REPO_ROOT).as_posix()} for r in all_runs]
                 ).to_csv(out / "runs.csv", index=False)
    print(f"\n{len(all_runs)} runs ({sum(r.checked for r in all_runs)} checked against their predictions); "
          f"tables in {out}")


if __name__ == "__main__":
    main()

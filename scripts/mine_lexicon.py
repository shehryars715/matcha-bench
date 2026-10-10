"""Step 3a (extension lexicon): mine disorder-specific candidate words from MATCHA training
users. Runs locally; nothing leaves the machine."""

import argparse

import _bootstrap  # noqa: F401
from matcha_bench.config import load_config, resolve_path
from matcha_bench.data import load_processed
from matcha_bench.lexicon.mine import mine_candidates


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/lexicon.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(resolve_path(args.config), args.set)
    m = cfg["mining"]

    import jieba

    jieba.setLogLevel(60)
    posts, _, _ = load_processed(resolve_path(cfg["processed_dir"]))
    cands = mine_candidates(posts, jieba.lcut, alpha0=m["alpha0"], z_min=m["z_min"], min_users=m["min_users"], top_k=m["top_k"])
    out = resolve_path(m["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    cands.to_csv(out, index=False, encoding="utf-8")
    print(cands.groupby("disorder").size().to_string())
    print(f"{len(cands)} candidates ({cands['word'].nunique()} unique words) -> {out}")


if __name__ == "__main__":
    main()

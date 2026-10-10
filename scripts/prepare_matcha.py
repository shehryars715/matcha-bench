"""Step 1: clean MATCHA, remove contaminated controls, assign global user-level splits."""

import argparse

import _bootstrap  # noqa: F401
from matcha_bench.config import load_config, matcha_root, resolve_path
from matcha_bench.data.matcha import prepare_matcha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. seed=1")
    args = ap.parse_args()
    cfg = load_config(resolve_path(args.config), args.set)
    stats = prepare_matcha(
        matcha_root(cfg.get("matcha_root")),
        resolve_path(cfg["processed_dir"]),
        seed=cfg["seed"],
        ratios=tuple(cfg["split_ratios"]),
        min_chars=cfg["min_chars"],
    )
    for key in ("raw_rows", "posts", "users", "contaminated_control_users", "multi_disorder_users", "ip_users"):
        print(f"{key:>28}: {stats[key]}")
    print(f"{'users_by_split':>28}: {stats['users_by_split']}")


if __name__ == "__main__":
    main()

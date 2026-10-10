"""Step 2: build downstream tasks (MATCHA-derived and Chinese MentalBERT) into data/tasks/<name>/."""

import argparse

import _bootstrap  # noqa: F401
from matcha_bench.config import load_config, resolve_path
from matcha_bench.tasks import acl_tasks, matcha_tasks, save_task


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/tasks.yaml")
    ap.add_argument("--only", nargs="*", help="build only these task names")
    ap.add_argument("--group", choices=["matcha", "acl", "all"], default="all")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(resolve_path(args.config), args.set)
    task_root, seed = resolve_path(cfg["task_root"]), cfg["seed"]

    def wanted(name):
        return not args.only or name in args.only

    def report(task):
        print(f"built {task.name:<32} " + " / ".join(f"{s}={len(df)}" for s, df in task.splits.items()))

    if args.group in ("matcha", "all") and any(wanted(n) for n in cfg["matcha"]):
        from matcha_bench.data import load_processed

        posts, users, ip = load_processed(resolve_path(cfg["processed_dir"]))
        for name, params in cfg["matcha"].items():
            if wanted(name):
                params = dict(params)
                builder = matcha_tasks.BUILDERS[params.pop("builder")]
                task = builder(posts, users, ip, name=name, seed=seed, **params)
                save_task(task, task_root)
                report(task)

    if args.group in ("acl", "all"):
        raw = resolve_path(cfg["acl_raw_dir"])
        for name in cfg["acl"]:
            if not wanted(name):
                continue
            try:
                task = acl_tasks.BUILDERS[name](raw, seed=seed)
            except FileNotFoundError as exc:
                print(f"skip  {name:<32} (missing {exc.filename})")
                continue
            save_task(task, task_root)
            report(task)


if __name__ == "__main__":
    main()

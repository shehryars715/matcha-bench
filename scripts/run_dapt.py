"""Step 5: domain-adaptive pre-training with random or lexicon-guided whole-word masking.

    python scripts/run_dapt.py --config configs/dapt/random.yaml
    python scripts/run_dapt.py --config configs/dapt/depression.yaml
    python scripts/run_dapt.py --config configs/dapt/deepseek.yaml
    python scripts/run_dapt.py --config configs/dapt/smoke.yaml      # CPU sanity check

Interrupted runs resume from the last checkpoint in output_dir automatically.
"""

import argparse

import _bootstrap  # noqa: F401
from matcha_bench.config import load_config, resolve_path
from matcha_bench.pretrain.run import run_dapt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="e.g. training.max_steps=100 seed=1")
    args = ap.parse_args()
    run_dapt(load_config(resolve_path(args.config), args.set))


if __name__ == "__main__":
    main()

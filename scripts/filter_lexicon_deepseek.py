"""Step 3b (extension lexicon): filter mined candidates with DeepSeek and write the
multi-disorder lexicon. Only candidate *words* are sent to the API.

    export DEEPSEEK_API_KEY=sk-...
    python scripts/filter_lexicon_deepseek.py --dry-run   # cost estimate, no API calls
    python scripts/filter_lexicon_deepseek.py
"""

import argparse

import pandas as pd

import _bootstrap  # noqa: F401
from matcha_bench.config import load_config, resolve_path
from matcha_bench.lexicon import save_lexicon
from matcha_bench.lexicon.deepseek import DeepSeekClient, estimate_cost, filter_with_deepseek


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/lexicon.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(resolve_path(args.config), args.set)
    ds = cfg["deepseek"]
    cands = pd.read_csv(resolve_path(cfg["mining"]["output"]))

    est = estimate_cost(len(cands), ds["votes"], ds["batch_size"], ds["price_per_m_input"], ds["price_per_m_output"])
    print(f"{len(cands)} (disorder, word) pairs x {ds['votes']} votes -> estimate {est}")
    if args.dry_run:
        return

    client = DeepSeekClient(model=ds["model"], base_url=ds["base_url"], temperature=ds["temperature"])
    ann = filter_with_deepseek(cands, client, resolve_path(ds["cache"]), votes=ds["votes"], batch_size=ds["batch_size"])
    ann.to_csv(resolve_path(ds["annotations"]), index=False, encoding="utf-8")

    kept = ann[(ann["keep"] == 1) & ~ann["category"].isin(ds.get("exclude_categories") or [])]
    save_lexicon(
        kept["word"].unique(),
        resolve_path(ds["output"]),
        header=f"MATCHA multi-disorder lexicon: log-odds mining + {ds['model']} filtering ({ds['votes']} votes)",
    )
    usage = client.usage
    cost = usage["prompt_tokens"] / 1e6 * ds["price_per_m_input"] + usage["completion_tokens"] / 1e6 * ds["price_per_m_output"]
    print(kept.groupby(["disorder", "category"]).size().unstack(fill_value=0).to_string())
    print(f"kept {kept['word'].nunique()} unique words; mean vote agreement {ann['agreement'].mean():.3f}")
    print(f"API usage this run: {dict(usage)} (~${cost:.3f})")


if __name__ == "__main__":
    main()

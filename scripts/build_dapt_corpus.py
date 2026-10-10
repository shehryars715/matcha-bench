"""Step 4: segment, tokenise and pack the DAPT corpus once (shared by all masking variants)."""

import argparse

import _bootstrap  # noqa: F401
from matcha_bench.config import load_config, resolve_path
from matcha_bench.data import CONTROL, load_processed
from matcha_bench.lexicon import load_lexicon
from matcha_bench.pretrain.corpus import build_corpus_cache, save_word_vocab
from matcha_bench.utils import save_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/corpus.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(resolve_path(args.config), args.set)

    posts, _, _ = load_processed(resolve_path(cfg["processed_dir"]))
    posts = posts[posts["source"].isin(cfg["sources"])]
    if not cfg["include_control"]:
        posts = posts[posts["cohort"] != CONTROL]
    if cfg["drop_reposts"]:
        posts = posts[~posts["is_repost"]]
    # A post can appear in several cohort files; pre-train on each note once, in time order.
    posts = posts.drop_duplicates("note_id").sort_values(["user_id", "created_at"])

    userdict: set = set()
    for path in cfg.get("userdict_lexicons") or []:
        p = resolve_path(path)
        if p.exists():
            userdict |= load_lexicon(p)
        else:
            print(f"warning: {p} not found. Build all lexicons first so segmentation is identical across variants.")

    train_texts = posts.loc[posts["split"].isin(cfg["train_splits"]), "text"].tolist()
    val_texts = posts.loc[posts["split"] == cfg["val_split"], "text"].tolist()[: cfg["max_val_texts"]]
    if cfg.get("max_texts"):
        train_texts = train_texts[: cfg["max_texts"]]

    out_dir = resolve_path(cfg["out_dir"])
    word2id: dict = {}
    common = dict(
        tokenizer_name=cfg["tokenizer"], word2id=word2id, max_seq_length=cfg["max_seq_length"],
        userdict_words=userdict, num_workers=cfg["num_workers"],
    )
    stats = {
        "train": build_corpus_cache(train_texts, out_dir / "train.npz", **common),
        "val": build_corpus_cache(val_texts, out_dir / "val.npz", **common),
        "userdict_words": len(userdict),
        "word_vocab": len(word2id),
    }
    save_word_vocab(word2id, out_dir / "words.json")
    save_json({**stats, "config": cfg}, out_dir / "corpus_stats.json")
    print(stats)


if __name__ == "__main__":
    main()

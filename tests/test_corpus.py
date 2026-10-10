import numpy as np

from matcha_bench.pretrain.corpus import ChunkDataset, build_corpus_cache, lexicon_lookup


def test_packing_respects_length_and_word_boundaries(tmp_path, tiny_tokenizer_dir, tiny_tokenizer):
    texts = ["我最近很绝望，想到死亡", "今天天气好", "失眠焦虑吃药", "医生说没事的"] * 5
    word2id: dict = {}
    stats = build_corpus_cache(
        texts, tmp_path / "train.npz", tokenizer_name=str(tiny_tokenizer_dir), word2id=word2id,
        max_seq_length=12, userdict_words=["绝望", "死亡", "失眠"],
    )
    assert stats["chunks"] > 1
    data = np.load(tmp_path / "train.npz")
    sizes = np.diff(data["offsets"])
    assert sizes.max() <= 10 and sizes.min() > 0
    # every chunk starts at a word boundary
    assert all(data["word_start"][o] == 1 for o in data["offsets"][:-1])
    assert "绝望" in word2id and "死亡" in word2id

    words = [""] * len(word2id)
    for w, i in word2id.items():
        words[i] = w
    lookup = lexicon_lookup(words, {"绝望", "死亡"})
    ds = ChunkDataset(tmp_path / "train.npz", tiny_tokenizer.cls_token_id, tiny_tokenizer.sep_token_id, lookup)
    ex = ds[0]
    assert ex["input_ids"][0] == tiny_tokenizer.cls_token_id and ex["input_ids"][-1] == tiny_tokenizer.sep_token_id
    assert len(ex["input_ids"]) == len(ex["word_start"]) == len(ex["lexicon"])
    decoded = tiny_tokenizer.convert_ids_to_tokens(ex["input_ids"])
    flagged = "".join(t for t, f in zip(decoded, ex["lexicon"]) if f)
    assert flagged in ("绝望死亡", "绝望", "死亡")

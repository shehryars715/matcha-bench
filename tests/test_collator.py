import random

import torch

from matcha_bench.pretrain.collator import LexiconGuidedWWMCollator, select_mask_positions, word_groups

# [CLS] 我 | 最近 | 很 | 绝望 | 想到 | 死亡 | 今天 | 天气 | 好 [SEP]  -> 17 tokens
WORD_START = [1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1]
LEXICON = [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  # 绝望, 死亡


def test_word_groups_exclude_specials():
    groups = word_groups(WORD_START)
    assert groups[0] == [1] and groups[1] == [2, 3]
    assert all(0 not in g and 16 not in g for g in groups)


def test_budget_is_twenty_percent_of_all_tokens():
    sel = select_mask_positions(WORD_START, [0] * 17, 0.2, rng=random.Random(0))
    assert sum(sel) <= round(17 * 0.2) and sum(sel) >= round(17 * 0.2) - 1


def test_lexicon_words_masked_first_as_whole_words():
    for seed in range(20):
        sel = select_mask_positions(WORD_START, LEXICON, 0.2, rng=random.Random(seed))
        # budget = round(3.4) = 3 -> 绝望 (2 tokens) fits, 死亡 would overflow; 1 slot left for a 1-token word
        assert sel[5] == sel[6] == 1
        assert sum(sel) <= 3
        assert sel[0] == sel[16] == 0


def test_whole_word_masking_never_splits_a_word():
    groups = word_groups(WORD_START)
    for seed in range(50):
        sel = select_mask_positions(WORD_START, [0] * 17, 0.3, rng=random.Random(seed))
        for g in groups:
            assert len({sel[i] for i in g}) == 1


def test_random_lexicon_order_varies():
    lex = [0] + [1] * 15 + [0]
    ws = [1] * 17
    picks = {tuple(select_mask_positions(ws, lex, 0.2, "random", random.Random(s))) for s in range(10)}
    assert len(picks) > 1


def test_collator_outputs(tiny_tokenizer):
    collator = LexiconGuidedWWMCollator(tiny_tokenizer, mlm_probability=0.2)
    ids = tiny_tokenizer("我最近很绝望想到死亡今天天气好")["input_ids"]
    assert len(ids) == 17
    ex = {"input_ids": ids, "word_start": WORD_START, "lexicon": LEXICON}
    torch.manual_seed(0)
    batch = collator([ex, {**ex, "input_ids": ids[:5] + ids[-1:], "word_start": [1] * 6, "lexicon": [0] * 6}])
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    assert batch["input_ids"].shape[1] % 8 == 0
    labels = batch["labels"][0]
    masked = (labels != -100).nonzero().flatten().tolist()
    assert 5 in masked and 6 in masked  # lexicon word 绝望 always selected
    assert torch.equal(labels[labels != -100], torch.tensor(ids)[masked])
    assert (batch["attention_mask"][1, 6:] == 0).all()

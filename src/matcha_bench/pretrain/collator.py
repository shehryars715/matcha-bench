"""Lexicon-guided whole-word masking (Zhai et al., 2024), reimplemented without string ops.

The reference implementation (``CustomDataCollatorForWholeWordMask`` in the Chinese
MentalBERT repository) works like this, and so does this collator:

1. Group tokens into whole words ([CLS]/[SEP] excluded).
2. Masking budget ``k = max(1, round(n_tokens * mlm_probability))``, where ``n_tokens``
   includes [CLS]/[SEP], with mlm_probability = 0.20.
3. Go through lexicon words in sentence order; mask each one whose tokens still fit in the
   budget.
4. Fill the rest of the budget with randomly ordered non-lexicon words, skipping any word
   that would overflow the budget.
5. Apply BERT corruption to the selected positions: 80% [MASK], 10% random token, 10% unchanged.

With an empty lexicon this reduces to plain whole-word masking at 20%, which is exactly the
"random masking" control in the paper. ``lexicon_order="random"`` shuffles lexicon words
instead of taking them in sentence order (an ablation the original paper does not run).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import torch


def word_groups(word_start: Sequence[int]) -> list[list[int]]:
    """Token index groups for whole words, excluding the first and last ([CLS]/[SEP]) positions."""
    groups: list[list[int]] = []
    for i in range(1, len(word_start) - 1):
        if word_start[i] or not groups:
            groups.append([i])
        else:
            groups[-1].append(i)
    return groups


def select_mask_positions(
    word_start: Sequence[int],
    lexicon: Sequence[int],
    mlm_probability: float = 0.2,
    lexicon_order: str = "sequential",
    rng: random.Random | None = None,
) -> list[int]:
    rng = rng or random
    n = len(word_start)
    budget = max(1, int(round(n * mlm_probability)))
    groups = word_groups(word_start)
    masked: set[int] = set()

    lex_groups = [g for g in groups if lexicon[g[0]]]
    if lexicon_order == "random":
        rng.shuffle(lex_groups)
    elif lexicon_order != "sequential":
        raise ValueError(f"lexicon_order must be 'sequential' or 'random', got {lexicon_order!r}")
    for g in lex_groups:
        if len(masked) + len(g) <= budget:
            masked.update(g)

    rest = [g for g in groups if g[0] not in masked]
    rng.shuffle(rest)
    for g in rest:
        if len(masked) >= budget:
            break
        if len(masked) + len(g) <= budget:
            masked.update(g)

    return [1 if i in masked else 0 for i in range(n)]


@dataclass
class LexiconGuidedWWMCollator:
    tokenizer: object
    mlm_probability: float = 0.2
    lexicon_order: str = "sequential"
    mask_replace_prob: float = 0.8
    random_replace_prob: float = 0.1
    pad_to_multiple_of: int | None = 8

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(e["input_ids"]) for e in examples)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m
        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.full((len(examples), max_len), pad_id, dtype=torch.long)
        attention = torch.zeros((len(examples), max_len), dtype=torch.long)
        mask_labels = torch.zeros((len(examples), max_len), dtype=torch.bool)
        for row, e in enumerate(examples):
            n = len(e["input_ids"])
            input_ids[row, :n] = torch.tensor(e["input_ids"], dtype=torch.long)
            attention[row, :n] = 1
            sel = select_mask_positions(e["word_start"], e["lexicon"], self.mlm_probability, self.lexicon_order)
            mask_labels[row, :n] = torch.tensor(sel, dtype=torch.bool)
        inputs, labels = self.corrupt(input_ids, mask_labels)
        return {"input_ids": inputs, "attention_mask": attention, "labels": labels}

    def corrupt(self, input_ids: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        labels = input_ids.clone()
        labels[~mask] = -100
        inputs = input_ids.clone()
        replace = torch.bernoulli(torch.full(labels.shape, self.mask_replace_prob)).bool() & mask
        inputs[replace] = self.tokenizer.mask_token_id
        remaining = 1.0 - self.mask_replace_prob
        p_random = self.random_replace_prob / remaining if remaining > 0 else 0.0
        randomise = torch.bernoulli(torch.full(labels.shape, p_random)).bool() & mask & ~replace
        inputs[randomise] = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)[randomise]
        return inputs, labels

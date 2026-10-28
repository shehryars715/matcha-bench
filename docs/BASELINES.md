# The three baselines: recipes and deviations

Each baseline is a published method reproduced on the MATCHA tasks. This page lists what each paper
did, what was run here, and every difference with its reason. Differences forced by the hardware
(one 16 GB GPU, Kaggle T4) are marked **HW**; they would go away on a larger GPU.

Shared by every model:
- the same user-level split, task files and full test sets; tuning on dev only, test scored once per run;
- macro-F1 as the primary metric on the MATCHA tasks;
- three training seeds (42, 43, 44), reported as mean ± SD (the source papers report single runs);
- fine-tuning per MATCHA task before testing, as each paper's protocol does;
- a paired bootstrap over the test set for every pair of models (`scripts/make_tables.py`: 10,000
  resamples shared by all models, difference in mean F1 over seeds, percentile 95% CI, two-sided p,
  Holm over the 36 pairs of a task);
- balanced class weights in the loss on the imbalanced tasks (`disorder_post(_masked)`,
  `user_multiclass_masked`) for the BERT-type models.

## 1. Chinese MentalBERT (Zhai et al., Findings of ACL 2024)

**Paper.** Domain-adaptive pre-training (DAPT) of `hfl/chinese-bert-wwm-ext` on 3.36M private Weibo
posts (mostly depression communities). Words from a depression lexicon (Li et al., 2020) are masked
first, random whole words fill the rest of a 20% budget. 3 epochs, batch 128, lr 5e-5, 128-token
chunks. Fine-tuning: lr 2e-5, batch 16, 10 epochs, best dev.

**Here.** The same recipe on MATCHA training users (`configs/dapt/`, `configs/finetune.yaml`):

| Model tag | What it is |
|---|---|
| `matcha-depression` | the recipe on MATCHA with the paper's lexicon: **the reproduction** |
| `matcha-random` | the same DAPT with random whole-word masking: isolates the lexicon's effect |
| `chinese-mentalbert` | the released model (`zwzzz/Chinese-MentalBERT`), fine-tuned the same way |
| `bert-wwm-ext` | the start model without DAPT |
| `matcha-multidisorder` | optional: a 520-word multi-disorder lexicon mined from MATCHA ([LEXICON.md](LEXICON.md)); sanity tasks only |

| | Paper | Here | Why |
|---|---|---|---|
| DAPT corpus | 3.36M private posts | MATCHA training users only (1.0M posts, 281k chunks) | the paper's corpus is not public; test users never seen |
| Segmentation | LTP | jieba primed with every lexicon | identical word boundaries for every variant |
| Masking budget | `round(0.2 · n_tokens)`, lexicon words first in sentence order, random top-up, 80/10/10 | same (`LexiconGuidedWWMCollator`, checked against the reference code) | — |
| DAPT on 2 GPUs | — | DataParallel; the logged loss is 2× and the grad-clip threshold effectively halved | same for every variant; true eval losses are recorded |
| Fine-tuning | 10 epochs, best dev | ≤ 10 epochs (≤ 6 on the user tasks), best dev, early stopping (patience 3; 2 on the user tasks) | selects on dev only |
| Fine-tuning batch | 16 | 16 on the MATCHA tasks; effective 32 on the sanity tasks (one run across two GPUs) | recorded per run as `effective_batch_size` |
| Max length | — | 128 tokens (posts), 512 (user timelines) | MATCHA posts are ≤ ~140 characters |
| Cognitive, Suicide | 5-fold cross-validation | one fixed split of the paper's sizes (2,180/545/682; 799/200/250) | one split for all models and seeds |
| Models on MATCHA | — | `bert-base-chinese` and `matcha-multidisorder` not run on the MATCHA tasks | GPU time (~8 GPU-h); both ran on the sanity tasks |

## 2. MentalGLM (Zhai et al., EMNLP 2025)

**Paper.** GLM-4-9B, LoRA instruction tuning in two stages; the second on the C-IMHI task data with
an explanation for every label, the three tasks trained together. 10 epochs, fp16, batch 4 × 8,
lr 1e-4, 1% warm-up, 1,024-token inputs, one V100 32 GB. Released: `zwzzz/MentalGLM-chat`.

**Here.** The released MentalGLM-chat, QLoRA-tuned per MATCHA task (`configs/llm.yaml`; tag
`mentalglm-chat-full`, and `mentalglm-chat` on `user_binary_masked`, where its 4,000-example and
500-dev caps exceed the task's 2,392 training and 358 dev users): batch 4 × 8, lr 1e-4, 1% warm-up,
LoRA rank 8 / alpha 16 on all projections (LLaMA-Factory defaults; the paper gives no values), every
training example, 2 epochs, best epoch on the full dev set.
The checkpoint is converted to the native transformers `glm` format
(`src/matcha_bench/llm/convert.py`; identical outputs to the original code).

| | Paper | Here | Why |
|---|---|---|---|
| Start model | GLM-4-9B → stage 1 → stage 2 | the released stage-2 model, MentalGLM-chat | the paper's best released model |
| Answers | label + explanation (explanations partly written by GPT-4) | label only | **the MATCHA DUA**: posts may not be sent to GPT-4 |
| Tasks | trained together | one adapter per task | matches the BERT protocol |
| Epochs | 10 | 2 | **HW**: ~5× the GPU time |
| Precision | fp16 LoRA | 4-bit QLoRA | **HW**: a 9B model in fp16 needs a ≥ 24–32 GB GPU; on the paper's own tasks 4-bit scores 84.13 / 67.25 vs fp16 82.95 / 69.56 |
| Input length | 1,024 tokens | 128 characters per post, 512 per timeline | about what the BERT models see (they cut at 128 / 512 tokens, ~1 token per Chinese character) |
| `disorder_post` seed 44 | — | diverged twice (NaN in fp16 activations at step ~175–220; the guard stopped it) → seed 45 | objective trigger fixed before the rerun, not chosen by score; both attempts are reported |
| Zero-shot rows | GLM-4-plus, GPT-3.5, GPT-4 | MentalGLM-chat and GLM-4-9B-chat, same prompt, 4-bit, greedy, first sentence parsed, unparsed = wrong | local models only (DUA); answer cap 24 tokens for MentalGLM-chat, 96 for GLM-4-9B-chat, whose answers open with a preamble |
| Label probabilities | — | `scripts/run_llm_scores.py`: one forward pass per item reads each label's probability where the answer names it (every label starts with its own token); checked against the generated test answers | an addition: generation gives hard labels only, and a class-prior correction needs probabilities on dev |

## 3. RoBERTa classifier (Murarka et al., LOUHI 2021)

**Paper.** `roberta-base` on Reddit posts (6 classes): Adam, lr 1e-5, 10 epochs, batch 16, dropout
0.3, 512 tokens, cross-entropy. F1 0.86 on posts.

**Here.** `hfl/chinese-roberta-wwm-ext` (Cui et al., 2021) with that recipe (`model_overrides` in
`configs/finetune.yaml`).

| | Paper | Here | Why |
|---|---|---|---|
| Model | `roberta-base` (English) | `hfl/chinese-roberta-wwm-ext` | the standard Chinese RoBERTa: BERT's architecture trained the RoBERTa way (no NSP, more data and steps) with whole-word masking |
| Loss | plain cross-entropy (balanced classes) | balanced class weights | MATCHA's classes are imbalanced (602 ADHD vs 5,000 DEP); the same weighting as every model |
| Input | posts (+ titles) | posts only, 512 tokens | Weibo has no titles; most MATCHA posts fit 128 tokens |
| Unstated settings | — | best dev epoch, linear decay without warm-up, weight decay 0, fp16, the standard HF head | as for every model |
| Not reproduced | LSTM and BERT baselines, CheckList tests | — | TF-IDF covers a non-pretrained baseline; the `_masked` tasks test label-word reliance |

## 4. Sanity checks

| Reproduction | Paper's task | Ours | Paper |
|---|---|---|---|
| Chinese MentalBERT (released model, fine-tuned, 3 seeds) | Suicide / Cognitive / Waimai / EWECT-epidemic | 81.52 / 73.90 / 85.52 / 65.86 | 86.15 / 74.75 / 89.70 / 67.77 |
| MentalGLM-chat (released, fp16, no tuning) | SOS-HL-1K / SocialCD-3K | 82.95 / 69.56 | 85.12 / 71.04 |
| RoBERTa-base (1 seed) | Reddit posts | 85.24 | 86 |

Every model we fine-tuned is 2.7–4.6 points below the Chinese MentalBERT paper on its two binary
tasks (Suicide, Waimai) and within about 2 points elsewhere; against the MentalGLM paper's numbers
for the same model on its own splits (81.30 / 74.91) the released Chinese MentalBERT matches. The
Reddit files released by Murarka et al. hold 16,931 posts, 1.3% fewer than the paper's 17,159.

## 5. Cut to save GPU time

| What | Saved |
|---|---|
| EWECT-usual stopped after 7 of 18 sanity runs | ~6 h |
| `matcha_ip_post` and the unmasked user-level tasks not run | ~20 h |
| Larger MentalGLM runs (10 epochs, fp16, 1,024 tokens) | open: needs a 24–32 GB GPU |

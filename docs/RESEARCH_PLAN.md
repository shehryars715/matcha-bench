# Research plan

This document records what the project is, the options that were considered, the decisions
that were made, and what remains to do.

---

## 0. The plan in two stages

1. **Stage 1: baselines (this repository).** Reproduce published methods on MATCHA and measure
   them on MATCHA tasks. MATCHA has no tasks or baselines of its own, so these scores are the
   reference point for anything that follows.
   - **1a. Chinese MentalBERT on MATCHA.** Lexicon-guided domain-adaptive pre-training (DAPT) on
     MATCHA, then fine-tuning on the MATCHA tasks. *In progress.*
   - **1b. An LLM baseline following MentalGLM.** LoRA instruction tuning of a ~9B Chinese LLM on the
     MATCHA tasks. *Planned; the exact setup is still to be decided (§3).*
2. **Stage 2: a new method.** The paper's contribution. It is evaluated against the stage-1
   baselines and will be designed once those numbers exist.

The Chinese MentalBERT paper's own evaluation tasks are kept as a **sanity check**: scores close to
the published ones show that the reproduction works. The **MATCHA tasks are the baselines**.

## 1. The source papers

**Chinese MentalBERT** (Zhai et al., Findings of ACL 2024, pp. 10574–10585) takes `hfl/chinese-bert-wwm-ext`
and continues masked-language-model pre-training on 3.36M Weibo posts, mostly depression
communities and the "Zoufan" tree hole. During that pre-training, words from a **depression
lexicon** (Li et al., 2020) are masked first, and random whole words fill the rest of a 20% budget.
It is evaluated on six tasks: EWECT-usual, EWECT-epidemic, Waimai, Cognitive (SocialCD-3K),
Suicide (SOS-HL-1K) and C2D2. It wins all six. Guided masking beats random masking by
+0.4 to +1.8 F1, on single runs with no variance reported. The corpus is not public.

**MentalGLM** (Zhai et al., same group; code at https://github.com/zwzzzQAQ/MentalGLM) fine-tunes
GLM-4-9B with LoRA in two steps: first on a Chinese translation of the English IMHI instruction
data, then on C-IMHI. C-IMHI covers Suicide, Cognitive and cognitive-pathway data, with an
explanation for every label: experts wrote examples, and GPT-4 wrote the rest. The model outputs a
label plus an explanation. Its table reports BERT baselines, prompted LLMs (GPT-4, GLM-4-plus) and
LoRA-tuned 7–8B LLMs. Chinese MentalBERT scores 81.30 F1 on SOS-HL-1K and 74.91 on SocialCD-3K;
MentalGLM-chat scores 85.12 and 71.04.

**MATCHA** (Li et al., TechRxiv 2025) is a Chinese Weibo dataset covering 7 DSM-5 disorders
plus controls: 1.81M posts and 41.8k users. It pairs disorder Super-Topic posts (STP) with
homepage timelines (HP), and a Qwen3 model marks "identified positive" users who explicitly
self-report. The paper is descriptive only (LIWC, symptoms, life events, a comparison with
Twitter-STMHD). It releases **no tasks, splits or baselines**.

## 2. Ways to "reproduce Chinese MentalBERT with MATCHA"

| # | Approach | What changes | Cost | Verdict |
|---|---|---|---|---|
| A | Faithful swap | MATCHA replaces the private corpus; same depression lexicon; same tasks | ~11 GPU-h DAPT (measured, 3 variants) | **kept**: `matcha-depression`, with `matcha-random` as the control |
| B | Multi-disorder lexicon (mined) | lexicon = log-odds over the 7 disorders vs control | CPU minutes | kept, as the input to C |
| C | Mined + LLM-filtered lexicon | B, filtered word by word by two LLM annotators | $0.26 | **kept** as an optional variant (`matcha-multidisorder`) |
| D | Pure LLM-generated lexicon | an LLM writes the words from scratch | < $0.10 | rejected: many words never occur in Weibo text, so they never trigger masking |
| E | Per-post LLM annotation as masking signal | an LLM tags every post | $40–150+, DUA risk | rejected: sends raw posts to an external API |
| F | Tasks on MATCHA | new detection tasks with leakage controls | fine-tuning GPU-h | **kept: these are the baselines** |
| G | LLM baseline | MentalGLM-style LoRA instruction tuning | heavy on a T4 | **planned as step 1b** |
| H | RoBERTa-wwm variants | second backbone | 2× GPU | deferred |
| I | Cross-lingual (Twitter-STMHD) | transfer test | needs STMHD access | future work |

The project was first framed as a benchmark paper. It was re-scoped on 2026-10-13: the tasks and
baselines are stage 1, and the paper's contribution will be a new method (stage 2).

## 3. Decisions taken

| Question | Decision |
|---|---|
| Compute | Kaggle with phone verification (free T4 ×2, 30 GPU-h/week). CPU-heavy steps run locally. |
| Scope | Stage 1 = baselines on MATCHA. Stage 2 = a new method (the paper). |
| Evaluation | 7 MATCHA tasks (the baselines; including masked variants) + the 5 public Chinese MentalBERT tasks (sanity check; C2D2 if obtained). |
| Models on the MATCHA tasks | TF-IDF, bert-base-chinese, chinese-bert-wwm-ext, Chinese MentalBERT, `matcha-random`, `matcha-depression` (the Chinese MentalBERT paper's comparison). `matcha-multidisorder` can be added later. |
| Models on the Chinese MentalBERT tasks | All of the above plus `matcha-multidisorder`. |
| Fine-tuning protocol | lr 2e-5, batch 16 per GPU, ≤10 epochs, early stopping (patience 3), best dev, 3 seeds, paired bootstrap. On Kaggle's 2 GPUs the effective batch is 32, identically for every model; to be stated in the paper. |
| Lexicon | Log-odds mining on MATCHA train users, then two LLM annotators; the 520-word consensus is kept. |
| LLM baseline (step 1b) | Planned. Open questions: our own LoRA tuning on MATCHA vs. the released model, whether to train with explanations (MATCHA posts cannot be sent to GPT-4 under the DUA), and which GPU to use. |
| Repo | Public code only; no data, word lists or model weights. |

## 4. LLM word filtering: options and cost

DeepSeek is a good fit. It is among the strongest models for Chinese, knows Weibo slang
(郁友, 焦友, emo了), supports JSON-mode output, and is cheap. Measured corpus size:
**1,882,257 post rows and 73.8M characters (about 50–75M tokens)**.

| Option | Sent to API | Estimated cost |
|---|---|---|
| Lexicon generation from knowledge only | disorder names | < $0.10 |
| **Hybrid (used)**: filter about 10k mined words × 3 votes | word lists only | **< $1** |
| Tag every post | 1.88M posts | ≈ $40 (V3-class) to $150+ (reasoning models), plus a likely DUA breach |

Prices were about $0.2–0.6 per 1M input tokens and $0.4–2.2 per 1M output tokens in September
2026, depending on the model. Check the official pricing page and update `configs/lexicon.yaml`.
`--dry-run` prints the estimate before anything is spent.

**Validation without human annotators.** No human annotators are available, so the lexicon is
validated in three ways:
- **Two independent LLM annotators.** The primary single-pass word filter (annotator A) is compared against
  DeepSeek (3 votes per (disorder, word), majority). Cohen's κ, percent agreement and the
  disagreement list come from `scripts/compare_lexicon_annotations.py`.
- **DeepSeek self-consistency.** The 3-vote agreement rate is logged.
- **Downstream evidence.** The share of the masking budget each lexicon fills is logged in
  `run_info.json`. An optional unfiltered-candidates DAPT variant would show whether the
  filtering itself matters.

The Limitations section should state that the lexicon was LLM-filtered and not human-verified.

## 5. Compute budget (Kaggle T4 ×2)

Measured on Kaggle (sessions 1–3, October 2026), except where marked as an estimate.

| Stage | Runs | Per run | Total |
|---|---|---|---|
| DAPT (281k chunks × 3 epochs, 6,594 steps at ~1.95 s/step) | 3 | 3.6 h | **10.8 h** (done) |
| Fine-tune: Chinese MentalBERT tasks, 6 models × 3 seeds | 90 | Suicide 1.9 min, Cognitive 5.7, Waimai 13.6, EWECT-epidemic 14.5, EWECT-usual 32 | ≈ 20 h (79 runs done) |
| Fine-tune: MATCHA tasks, 5 models × 3 seeds | 105 | estimated: ip_post ~32 min, disorder_post ~38, user_binary ~12, user_multiclass ~35 (each task also has a masked twin) | ≈ 50 h (estimate) |
| **Total** | | | **≈ 80 GPU-h ≈ 3 weeks of Kaggle quota** |

Ways to cut it, from least to most costly to the results:
- drop the leftover `matcha-multidisorder` / `bert-base-chinese` runs on the sanity-check tasks;
- drop or postpone `matcha_ip_post`, which mostly reflects how MATCHA was collected;
- keep only the masked versions of the user-level tasks;
- fewer seeds or fewer models (not recommended: model gaps are small, so 3 seeds and the
  random-masking control are needed to interpret them).

CPU-only steps can run on a laptop: data preparation, task building, mining, corpus packing and TF-IDF.

## 6. Questions the baselines answer

1. **Corpus effect:** `matcha-random` vs `bert-wwm-ext`. Does pre-training on MATCHA help?
2. **Lexicon effect (the Chinese MentalBERT claim):** `matcha-depression` vs `matcha-random`. Does
   guided masking help on an open corpus, with seeds and significance tests?
3. **Corpus comparison:** `matcha-depression` vs `chinese-mentalbert` (private depression corpus).
4. **Shortcut reliance:** every model, raw vs `_masked`. The size of the drop shows how much is
   keyword matching. Does DAPT reduce that reliance?
5. **Multi-disorder lexicon (optional):** `matcha-multidisorder` vs `matcha-depression`, so far on
   the Chinese MentalBERT tasks only.
6. **Where every baseline struggles** (per-disorder and per-task results). This tells stage 2 where
   a new method has the most room.

### Preliminary observations

**Local CPU runs (2026-10-11)**
- **Cleaning:** 1,882,257 raw rows → 1,507,148 posts (1,439,838 unique notes) from 41,747 users.
  766 control users also post in disorder communities and were removed from the control group.
  2,677 users belong to more than one disorder cohort.
- **Chinese MentalBERT task splits** match the paper exactly for Waimai (9,589/1,199/1,199) and Cognitive
  (2,180/545/682), and within 1% for EWECT and Suicide.
- **The depression lexicon covers little of MATCHA.** 601 of its 1,470 words occur in a 20k-post
  sample. Only 43% of 128-token chunks contain a lexicon word, and lexicon words fill only
  **≈ 8% of the masking budget** (10.3% on the full corpus); the rest is random top-up.
- **Mining:** 9,307 (disorder, word) candidates, 4,269 unique.
- **Multi-disorder lexicon (see `docs/LEXICON.md`):** annotator A kept 526 words, DeepSeek kept 1,439
  (3 votes, $0.26). The consensus lexicon (both keep) has **520 words**. Word-level κ = 0.43,
  pair-level κ = 0.56, and 98.9% of A's words were confirmed by DeepSeek. On the full corpus it fills
  **17.3% of the masking budget vs 10.3%** for the depression lexicon (62.8% vs 53.4% of chunks).
- **Shortcut effect (TF-IDF baseline):** `matcha_disorder_post` macro-F1 is **48.8**, but only **39.6**
  once disorder and drug names are masked. Keywords carry a large share of the signal.
- TF-IDF on the Chinese MentalBERT tasks: Suicide F1 75.4, Cognitive micro-F1 63.1. The paper reports
  Word2vec-CNN at 82.4 and 68.6.

**Sanity check on Kaggle (2026-10-12; test F1, mean of 3 seeds)**

| Model | Suicide | Cognitive (micro) | Waimai |
|---|---|---|---|
| bert-base-chinese | 79.8 | 71.9 | 84.3 |
| bert-wwm-ext | 80.6 | 73.7 | 84.8 |
| chinese-mentalbert | 81.5 | 73.9 | 85.5 |
| matcha-random | 81.6 | 74.3 | 84.2 |
| matcha-depression | 81.9 | 74.6 | 84.4 |
| matcha-multidisorder | 80.6 | 73.6 | 84.7 |

- Our Chinese MentalBERT scores are close to those in MentalGLM's table (81.30 / 74.91), so the
  fine-tuning setup reproduces. Those splits differ slightly from ours.
- MATCHA pre-training gives +0.6 to +1.0 F1 over `bert-wwm-ext` on the mental-health tasks. The lexicon
  adds about +0.3 over random masking, mostly within seed variation (depression beats random on all
  3 Suicide seeds).
- Waimai, the non-mental-health control, barely moves.

## 7. Paper outline (provisional)

The paper's contribution is the stage-2 method. The baselines enter as setup and comparison.

1. Introduction: multi-disorder detection on Chinese social media; MATCHA has no tasks or baselines.
2. Related work: mental-health PLMs and LLMs, Chinese resources, shortcut learning (Harrigian et al., 2020).
3. Tasks: MATCHA cleaning, contamination, user-level splits, tasks and masked variants.
4. Baselines: Chinese MentalBERT reproduced on MATCHA (with the sanity check), the LLM baseline, TF-IDF.
5. Method (stage 2).
6. Experiments: method vs baselines, seeds and significance tests.
7. Analysis: per-disorder results, raw vs masked, fairness.
8. Limitations and ethics.

## 8. To do

- [ ] Finish the stage-1 fine-tuning on Kaggle (`python scripts/kaggle_pipeline.py start`).
- [ ] Decide which runs to cut, if any (§5).
- [ ] Decide on the effective fine-tuning batch: keep 32 and state it, or re-run with 16.
- [ ] Step 1b: settle the LLM-baseline setup (§3) and the GPU to run it on.
- [ ] Confirm with the MATCHA authors whether the DUA allows (a) a private Kaggle dataset copy,
      (b) releasing split IDs, and (c) releasing the mined lexicon.
- [ ] Request C2D2 from its authors to complete the six Chinese MentalBERT tasks.
- [x] Second-annotator check of the multi-disorder lexicon with DeepSeek (Cohen's κ; `docs/LEXICON.md`).
- [ ] Optional: a human spot-check, if a Chinese-speaking annotator becomes available.
- [ ] State in the paper which LLMs did the word filtering (model names and date).
- [ ] Release DAPT model weights only if the DUA permits it.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Guided masking gives no gain over random | Still a useful baseline finding, backed by seeds and significance tests. |
| MATCHA tasks are trivially easy (keywords) | The masked variants and user-level HP-only tasks are designed to expose this. |
| Small classes (ADHD: 32 test users) | Report per-class F1 with CIs; macro-F1 as primary. |
| Kaggle limits (12 h per session, 30 GPU-h per week) | Sessions stop cleanly and resume; the cut options in §5; a local GPU for the LLM baseline. |
| DUA restrictions | No post leaves the machine; only k-anonymous words are sent to the API. |

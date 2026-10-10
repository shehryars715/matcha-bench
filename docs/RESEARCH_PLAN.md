# Research plan

This document records what the project is, the options that were considered, the decisions
that were made, and what remains to do before a submission.

---

## 1. The two source papers in one paragraph each

**Chinese MentalBERT** (Zhai et al., Findings of ACL 2024, pp. 10574–10585) takes `hfl/chinese-bert-wwm-ext`
and continues masked-language-model pre-training on 3.36M Weibo posts, mostly depression
communities and the "Zoufan" tree hole. During that pre-training, words from a **depression
lexicon** (Li et al., 2020) are masked first, and random whole words fill the rest of a 20% budget.
It is evaluated on six tasks: EWECT-usual, EWECT-epidemic, Waimai, Cognitive (SocialCD-3K),
Suicide (SOS-HL-1K) and C2D2. It wins all six. Guided masking beats random masking by
+0.4 to +1.8 F1, on single runs with no variance reported. The corpus is not public.

**MATCHA** (Li et al., TechRxiv 2025) is a Chinese Weibo dataset covering 7 DSM-5 disorders
plus controls: 1.81M posts and 41.8k users. It pairs disorder Super-Topic posts (STP) with
homepage timelines (HP), and a Qwen3 model marks "identified positive" users who explicitly
self-report. The paper is descriptive only (LIWC, symptoms, life events, a comparison with
Twitter-STMHD). It releases **no tasks, splits or baselines**.

## 2. Every way to "reproduce Chinese MentalBERT with MATCHA"

| # | Approach | What changes | Novelty | Cost | Verdict |
|---|---|---|---|---|---|
| A | Faithful swap | MATCHA replaces the private corpus; same depression lexicon; same 6 tasks | low: a reproduction study | ~6 GPU-h DAPT | **kept**, as the reproduction arm |
| B | Multi-disorder lexicon (mined) | lexicon = log-odds over the 7 disorders vs control | medium | + CPU minutes | kept, as the input to C |
| C | **Mined + LLM-filtered lexicon** | B, filtered word by word by an LLM annotator (DeepSeek script kept as an alternative) | medium–high | free–$1 | **kept, the extension** |
| D | Pure LLM-generated lexicon | an LLM writes the words from scratch | medium | < $0.10 | rejected: many words never occur in Weibo text, so they never trigger masking |
| E | Per-post LLM annotation as masking signal | an LLM tags every post | high | $40–150+, DUA risk | rejected: sends raw posts to an external API |
| F | Benchmark on MATCHA | new tasks with leakage controls | **high**, a first for MATCHA | fine-tuning GPU-h | **kept as the primary contribution** |
| G | LLM baselines (Qwen-7B LoRA) | like the paper's Llama2/Alpaca rows | medium | heavy on a T4 | deferred (optional section) |
| H | RoBERTa-wwm variants | second backbone | low | 2× GPU | deferred |
| I | Cross-lingual (Twitter-STMHD) | transfer test | high | needs STMHD access | future work |

**Chosen framing (benchmark-first):** *"MATCHA-Bench: a leakage-aware benchmark for multi-disorder
mental health detection on Chinese social media, with a reproduction and multi-disorder
extension of lexicon-guided domain-adaptive pre-training."*

## 3. Decisions taken

| Question | Decision |
|---|---|
| Compute | Kaggle with phone verification (free T4/P100, about 30 GPU-h/week). CPU-heavy steps run locally. |
| Paper focus | Benchmark first; DAPT variants are a model family inside it. |
| Evaluation | The public Chinese MentalBERT tasks plus 7 MATCHA tasks (including masked variants). |
| Lexicon | Hybrid: log-odds mining on MATCHA train users, then single-pass LLM filtering of the words (no API key needed). The DeepSeek 3-vote filter remains available. |
| Models | TF-IDF, bert-base-chinese, chinese-bert-wwm-ext, Chinese MentalBERT, and 3 MATCHA-DAPT variants. |
| Repo | Local git first; pushed to GitHub after review. |

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

## 5. Compute budget (Kaggle T4)

These are estimates; replace them with timings from the first real run.

| Stage | Runs | Per run | Total |
|---|---|---|---|
| DAPT (1.01M train posts = 35.3M tokens = 281k chunks × 3 epochs, measured) | 3 | 1–2 h | 3–6 h |
| Fine-tune: Chinese MentalBERT tasks (5) | 6 models × 3 seeds | ≈ 45 min | ≈ 14 h |
| Fine-tune: MATCHA tasks (7) | 6 models × 3 seeds | ≈ 1–1.5 h | ≈ 20–25 h |
| **Total** | | | **≈ 40–50 GPU-h ≈ 2 weeks of Kaggle quota** |

Ways to cut it:
- run baselines with 1 seed first;
- prioritise `cognitive`, `suicide` and the masked MATCHA tasks;
- use both T4s (handled automatically).

CPU-only steps can run on a laptop: data preparation, task building, mining, corpus packing and TF-IDF.

## 6. Experiment matrix and expected findings

1. **Corpus effect:** `matcha-random` vs `bert-wwm-ext`. Expect gains on the mental-health tasks.
2. **Lexicon effect (reproduction):** `matcha-depression` vs `matcha-random`. Does the Chinese MentalBERT claim hold on an
   open corpus with seeds and significance tests?
3. **Multi-disorder lexicon:** `matcha-multidisorder` vs `matcha-depression`. Expect the largest gap on
   non-depression MATCHA tasks (BD, OCD, SZ per-class F1).
4. **Shortcut reliance:** every model, raw vs `_masked`. The size of the drop shows how much is keyword
   matching. Does DAPT reduce that reliance?
5. **Cross-corpus check:** `chinese-mentalbert` (private corpus) vs `matcha-*` on the Chinese MentalBERT tasks.
6. **Analysis:** lexicon coverage, the category mix of the multi-disorder lexicon, fill-mask probes on
   SCL-90 items (as in the Chinese MentalBERT paper's Table 5), per-gender error rates.

### Preliminary observations (local CPU runs, 2026-10-11)

These come from the first local runs. They are not final results.

- **Cleaning:** 1,882,257 raw rows → 1,507,148 posts (1,439,838 unique notes) from 41,747 users.
  766 control users also post in disorder communities and were removed from the control group.
  2,677 users belong to more than one disorder cohort.
- **Chinese MentalBERT task splits** match the paper exactly for Waimai (9,589/1,199/1,199) and Cognitive
  (2,180/545/682), and within 1% for EWECT and Suicide.
- **The depression lexicon covers little of MATCHA.** 601 of its 1,470 words occur in a 20k-post
  sample. Only 43% of 128-token chunks contain a lexicon word, and lexicon words fill only
  **≈ 8% of the masking budget**; the rest is random top-up. This may explain the small
  guided-vs-random gains in the Chinese MentalBERT paper, and it motivates a larger, multi-disorder lexicon.
- **Mining:** 9,307 (disorder, word) candidates, 4,269 unique.
- **Multi-disorder lexicon (see `docs/LEXICON.md`):** annotator A kept 526 words, DeepSeek kept 1,439
  (3 votes, $0.26). The consensus lexicon (both keep) has **520 words**. Word-level κ = 0.43,
  pair-level κ = 0.56, and 98.9% of A's words were confirmed by DeepSeek. On the full corpus it fills
  **17.3% of the masking budget vs 10.3%** for the depression lexicon (62.8% vs 53.4% of chunks).
- **Shortcut effect (TF-IDF baseline):** `matcha_disorder_post` macro-F1 is **48.8**, but only **39.6**
  once disorder and drug names are masked. Keywords carry a large share of the signal.
- TF-IDF on the Chinese MentalBERT tasks: Suicide F1 75.4, Cognitive micro-F1 63.1. The paper reports
  Word2vec-CNN at 82.4 and 68.6.

## 7. Paper outline (target: ACL/EMNLP Findings or the CLPsych workshop)

1. Introduction: no multi-disorder Chinese benchmark; DAPT gains unverified on open data.
2. Related work: mental-health PLMs, Chinese resources, shortcut learning (Harrigian et al., 2020).
3. MATCHA-Bench: cleaning, contamination, splits, tasks, masked variants, statistics.
4. Lexicon-guided DAPT: the method, the depression lexicon, multi-disorder mining and LLM filtering.
5. Experimental setup: models, protocol, seeds, significance testing.
6. Results: Chinese MentalBERT tasks (reproduction), MATCHA tasks, raw vs masked.
7. Analysis: lexicon coverage, per-disorder results, fill-mask probes, fairness.
8. Limitations and ethics.

## 8. Before submitting

- [ ] Confirm with the MATCHA authors whether the DUA allows (a) a private Kaggle dataset copy,
      (b) releasing split IDs, and (c) releasing the mined lexicon.
- [ ] Request C2D2 from its authors to complete the six Chinese MentalBERT tasks.
- [x] Second-annotator check of the multi-disorder lexicon with DeepSeek (Cohen's κ; `docs/LEXICON.md`).
- [ ] Optional: a human spot-check, if a Chinese-speaking annotator becomes available.
- [ ] State in the paper which LLM did the word filtering (model name and date).
- [ ] Replace the compute estimates in §5 with measured times.
- [ ] Optional: a Qwen2.5-7B LoRA baseline on the post-level tasks.
- [ ] Release DAPT model weights only if the DUA permits it.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Guided masking gives no gain over random | Still publishable: this is a negative reproduction backed by seeds and significance tests. |
| MATCHA tasks are trivially easy (keywords) | The masked variants and user-level HP-only tasks are designed to expose this. |
| Small classes (ADHD: 32 test users) | Report per-class F1 with CIs; macro-F1 as primary. |
| Kaggle session limits (12 h) | DAPT auto-resumes from checkpoints; fine-tuning skips finished runs. |
| DUA restrictions | No post leaves the machine; only k-anonymous words are sent to the API. |

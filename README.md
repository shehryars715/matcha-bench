<div align="center">

# MATCHA-Bench

**Reproducible baselines for multi-disorder mental health detection on Chinese social media:<br>
Chinese MentalBERT, MentalGLM and a RoBERTa classifier, reproduced on MATCHA**

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.46%2B%20%7C%205.x-FFD21E.svg)](https://github.com/huggingface/transformers)
[![Code license: MIT](https://img.shields.io/badge/code%20license-MIT-green.svg)](LICENSE)
[![Data: DUA required](https://img.shields.io/badge/data-DUA%20required-lightgrey.svg)](data/README.md)

</div>

---

## Overview

[MATCHA](https://doi.org/10.36227/techrxiv.176685818.81296428/v1) (Li et al., 2025) is the first
**multi-disorder** Chinese Weibo dataset: 1.8M posts from 41.8k users across seven DSM-5 disorders
(ADHD, ASD, bipolar, depression, GAD, OCD, schizophrenia) plus controls. It is descriptive only: it
releases no tasks, splits or baselines.

This repository defines **four leakage-aware detection tasks** on MATCHA and reproduces **three
published methods** on them. Their scores are the reference point for new methods on MATCHA; a new
method is planned as follow-up work.

| Baseline | Paper | What is reproduced here |
|---|---|---|
| Chinese MentalBERT | Zhai et al., [Findings of ACL 2024](https://aclanthology.org/2024.findings-acl.629/) | lexicon-guided domain-adaptive pre-training (DAPT) of `hfl/chinese-bert-wwm-ext` on MATCHA, plus the released model, the start model and a random-masking control |
| MentalGLM | Zhai et al., [EMNLP 2025](https://aclanthology.org/2025.emnlp-main.686/) | the released MentalGLM-chat (GLM-4-9B), QLoRA-tuned per task with label-only answers; zero-shot rows as an extra |
| RoBERTa classifier | Murarka et al., LOUHI 2021 ([code](https://github.com/amurark/mental-health-detection)) | `hfl/chinese-roberta-wwm-ext` with the paper's recipe |

Each reproduction is first checked on its paper's own evaluation tasks (the sanity checks below).
Every deviation from the papers, and why, is listed in [`docs/BASELINES.md`](docs/BASELINES.md).

## Results

Test macro-F1 (%) on the full test sets, mean ± SD over three training seeds (42, 43, 44; one run for
TF-IDF and the zero-shot rows). Bold: best trained baseline per task. Built by
`python scripts/make_tables.py` from the finished runs.

| Model | `disorder_post` | `disorder_post_masked` | `user_binary_masked` | `user_multiclass_masked` |
|---|---|---|---|---|
| TF-IDF + LR | 48.83 | 39.56 | 90.14 | 41.68 |
| Chinese-BERT-wwm-ext (start model) | 51.57 ± 0.84 | 42.31 ± 1.49 | 91.64 ± 0.60 | 49.25 ± 1.51 |
| Chinese MentalBERT (released) | 51.62 ± 3.34 | 43.70 ± 1.98 | 90.48 ± 0.88 | 50.26 ± 1.02 |
| DAPT on MATCHA, random masking (control) | 51.51 ± 2.62 | 45.23 ± 1.94 | 91.49 ± 1.02 | 51.49 ± 0.98 |
| DAPT on MATCHA, lexicon-guided (Chinese MentalBERT recipe) | 51.26 ± 2.06 | 44.87 ± 1.88 | **92.08 ± 0.73** | 52.27 ± 0.35 |
| RoBERTa-wwm-ext (Murarka et al. recipe) | 50.11 ± 0.59 | 41.76 ± 0.91 | 91.11 ± 0.59 | 50.65 ± 0.56 |
| MentalGLM-chat, QLoRA fine-tuned | **60.81 ± 3.14** | **51.42 ± 1.05** | 88.98 ± 0.25 | **54.83 ± 0.83** |
| MentalGLM-chat, zero-shot | 41.58 | 29.74 | 49.52 | 20.74 |
| GLM-4-9B-chat, zero-shot (control) | 31.34 | 17.34 | 19.41 | 6.87 |

Test sets: 20,000 posts (`disorder_post`, `_masked`), 690 users (`user_binary_masked`), 2,105 users
(`user_multiclass_masked`). MentalGLM's `disorder_post` cell uses seeds 42, 43 and 45: seed 44
diverged in fp16 twice and was replaced by the next unused seed, a rule fixed before the rerun.

Significance: a paired bootstrap with 10,000 resamples of each test set, the same resamples for
every model, comparing the mean F1 over seeds; two-sided p, Holm-corrected over the 36 model pairs
of each task (`outputs/results/paper/bootstrap.md`).

- **The fine-tuned LLM leads on posts.** MentalGLM-chat beats every other trained model and TF-IDF by
  9.2–12.0 points on `disorder_post` and 6.2–11.9 on `disorder_post_masked` (Holm p < 0.01 for every
  pair). On
  `user_multiclass_masked` it is 2.6–13.2 points ahead, significant against TF-IDF, the start model
  and the released Chinese MentalBERT, but not against the MATCHA-pre-trained models or RoBERTa. On
  `user_binary_masked` it is 1.5–3.1 points *behind* the BERT-type models (significant only against
  the lexicon-guided model).
- **Pre-training on MATCHA helps once disorder names are hidden.** On `disorder_post_masked` both
  MATCHA DAPT models beat the start model (+2.6 and +2.9) and the released Chinese MentalBERT
  (+1.2 and +1.5); on `user_multiclass_masked` the lexicon-guided model beats the start model
  (+3.0). With names visible, all BERT models tie.
- **Lexicon-guided masking adds nothing measurable over random masking** (|Δ| ≤ 0.8 on every task,
  p ≥ 0.28), the Chinese MentalBERT paper's central claim.
- **RoBERTa does not beat BERT here.** It is 1.1–1.5 points below every BERT model on
  `disorder_post` and 1.9–3.5 below the released and MATCHA-pre-trained models on
  `disorder_post_masked`; the user tasks tie.
- **Much of the unmasked score is keyword spotting.** Hiding disorder and drug names costs 6–9.5
  points for every model on `disorder_post` (TF-IDF 48.8 → 39.6).
- **Zero-shot LLMs are far behind.** GLM-4-9B-chat names a label in only 23–44% of its answers;
  MentalGLM-chat in 99% or more.

### Sanity checks (each reproduction on its paper's own tasks)

| Reproduction | Task | Ours | Paper |
|---|---|---|---|
| Chinese MentalBERT, released model, fine-tuned (3 seeds) | Suicide (SOS-HL-1K, binary F1) | 81.52 ± 0.24 | 86.15 (Chinese MentalBERT paper); 81.30 (MentalGLM paper) |
| | Cognitive (SocialCD-3K, micro-F1) | 73.90 ± 0.82 | 74.75; 74.91 |
| | Waimai (binary F1) | 85.52 ± 0.11 | 89.70 |
| | EWECT-epidemic (macro-F1) | 65.86 ± 0.59 | 67.77 |
| MentalGLM-chat (released), fp16, no tuning | Suicide | 82.95 | 85.12 |
| | Cognitive | 69.56 | 71.04 |
| RoBERTa-base, fine-tuned (1 seed, as in the paper) | Reddit posts (6 classes, macro-F1) | 85.24 | 86 |

Against the Chinese MentalBERT paper, every model we fine-tuned (BERT, wwm-ext and the released
model alike) is 2.7–4.6 points lower on the two binary tasks (Suicide, Waimai) and within about 2
points elsewhere; as in the paper, the released model is the best of the three on both binary tasks. The paper's
Suicide and Cognitive numbers come from 5-fold cross-validation, ours from one fixed split of the
same sizes. Full tables: `outputs/results/paper/sanity_*.md`.

## Reproduce

### What you need

- **MATCHA access** under its Data Use Agreement ([`data/README.md`](data/README.md)). No data,
  derived post-level files or lexicons are included here.
- Python 3.10+ and `pip install -e ".[dev,llm]"` (or `pip install -r requirements.txt`, plus
  `peft bitsandbytes` for MentalGLM). The CPU steps were run on Windows (Python 3.14,
  transformers 5.9); the GPU steps on Kaggle's GPU image of October 2026 (Python 3.13, torch 2.11,
  transformers 5.16, peft 0.20, bitsandbytes 0.50). Exact image digests and a full package list:
  [`docs/environment/`](docs/environment/README.md).
- One NVIDIA GPU with 16 GB (a T4 is enough for everything; DAPT ran on two).
- For the optional multi-disorder lexicon only: a DeepSeek API key in `.env` (about $0.26).

### Measured cost

| Step | Hardware | Measured time |
|---|---|---|
| Data preparation, tasks, lexicons, DAPT corpus | CPU | ~20 min |
| Chinese MentalBERT: DAPT, per variant (random, lexicon-guided; multi-disorder optional) | 2 × T4 | 3.6 h |
| Chinese MentalBERT: fine-tuning on the MATCHA tasks, 4 models × 4 tasks × 3 seeds | 1 × T4 | ~13 GPU-h (per run: 20 min post tasks, 6 binary, 17 multiclass) |
| Chinese MentalBERT: sanity tasks, 6 models × 4 tasks × 3 seeds | 2 × T4 | ~11 h (per run: Suicide 2 min, Cognitive 6, Waimai 14, EWECT-epidemic 15) |
| MentalGLM: sanity check, released model | 1 × T4 | ~1 h |
| MentalGLM: fine-tuning, 4 tasks × 3 seeds | 1 × T4 | ~100 GPU-h (per run: 11.3 h post tasks, 8 h multiclass, 2.5 h binary) |
| MentalGLM: zero-shot rows, 2 models × 4 tasks | 1 × T4 | ~5.4 h |
| RoBERTa: 4 tasks × 3 seeds, plus the Reddit sanity check | 1 × T4 | ~5.4 GPU-h + 1 h |
| Tables and bootstrap | CPU | ~3 min |

### Step by step

The notebooks in [`notebooks/`](notebooks/) walk through each step with the expected outputs:

| Notebook | Step |
|---|---|
| [`01_data_split_tasks.ipynb`](notebooks/01_data_split_tasks.ipynb) | clean MATCHA, user-level split, build all tasks |
| [`02_lexicon.ipynb`](notebooks/02_lexicon.ipynb) | depression lexicon; optional multi-disorder lexicon |
| [`03_chinese_mentalbert.ipynb`](notebooks/03_chinese_mentalbert.ipynb) | DAPT corpus, DAPT, fine-tuning, sanity check |
| [`04_mentalglm.ipynb`](notebooks/04_mentalglm.ipynb) | checkpoint conversion, sanity check, QLoRA fine-tuning, zero-shot |
| [`05_roberta.ipynb`](notebooks/05_roberta.ipynb) | RoBERTa on MATCHA and the Reddit sanity check |
| [`06_results.ipynb`](notebooks/06_results.ipynb) | the results tables and the bootstrap |

The same steps from the command line:

```bash
git clone https://github.com/shehryars715/matcha-bench.git && cd matcha-bench
pip install -e ".[dev,llm]"
python -m pytest -q                               # offline unit tests

# 1. data (MATCHA under its DUA: data/README.md)
python scripts/prepare_matcha.py                  # clean, dedup, remove contaminated controls, user split
python scripts/split_fingerprint.py               # compare with the fingerprints below
python scripts/download_mentalbert_data.py        # the Chinese MentalBERT evaluation datasets
python scripts/build_tasks.py                     # every task -> data/tasks/

# 2. lexicons
python scripts/build_depression_lexicon.py        # the paper's depression lexicon (1,470 words)
# optional multi-disorder lexicon (notebooks/02_lexicon.ipynb), then configs/pipeline_multidisorder.yaml

# 3. baselines (GPU); each pipeline skips finished work and resumes after an interruption
python scripts/build_dapt_corpus.py
python scripts/run_pipeline.py --no-limit --config configs/pipeline_mentalbert.yaml
python scripts/run_pipeline.py --no-limit --config configs/pipeline_mentalglm.yaml
python scripts/run_pipeline.py --no-limit --config configs/pipeline_roberta.yaml

# 4. tables + 10,000-resample paired bootstrap -> outputs/results/paper/
python scripts/make_tables.py
```

**Split fingerprints.** The split is rebuilt deterministically (seed 42). `split_fingerprint.py`
prints the SHA-256 of each split's sorted user IDs; ours are

| Split | Users | SHA-256 |
|---|---|---|
| train | 29,222 | `400c57e3fd6003a3443feef7d23a9c640e08f5af1ca64808eedc35ff670cde52` |
| dev | 4,175 | `aac84d29a48e2c971b4a63d6b4a264479b19ab7985622de8a3aae674a5a55410` |
| test | 8,350 | `44934b89108b9d7a049711b6a59fdb229fb63e3006ec69e3ddaaa671fc255042` |

The ID lists themselves (`--write-ids`) are Weibo user IDs and stay under the DUA.

**How the reported runs were made.** On Kaggle's free T4 GPUs, in resumable sessions spread over
several accounts; that tooling lives in `scripts/kaggle/` and `configs/kaggle/` and is described in
[`docs/KAGGLE.md`](docs/KAGGLE.md). It is not needed to reproduce the results.

## Tasks

MATCHA defines no tasks, so these are designed here. One **user-level** split (70/10/20, stratified
by cohort, seed 42) is shared by every task and by the DAPT corpus, so no person appears on both
sides and pre-training sees training users only.

| Task | Level | Labels | Metric | Question |
|---|---|---|---|---|
| `matcha_disorder_post` | post | 7 | macro-F1 | which disorder community does the post come from? |
| `matcha_disorder_post_masked` | post | 7 | macro-F1 | the same, with disorder and drug names hidden |
| `matcha_user_binary_masked` | user | 2 | macro-F1 | disorder vs control, from **homepage posts only**, names hidden |
| `matcha_user_multiclass_masked` | user | 8 | macro-F1 | 7 disorders + control, from **homepage posts only**, names hidden |

Also built, but not part of the baselines: `matcha_user_binary` / `matcha_user_multiclass` (without
masking) and `matcha_ip_post` (is a post an explicit self-report of diagnosis or treatment?).

**Leakage controls.**
- 766 "control" users who also post in disorder communities are removed from the control group.
- User-level tasks use homepage timelines only, because Super-Topic posts name the disorder.
- `_masked` variants replace disorder names, community nicknames and 60+ psychiatric drug names
  with placeholders (`某病`, `某药`).
- `disorder_post` training is capped at 5,000 posts per disorder (23,109 posts) and its test set at
  20,000 posts, stratified.

## Repository layout

```
├── configs/
│   ├── pipeline_{mentalbert,mentalglm,roberta}.yaml   one pipeline per baseline
│   ├── finetune.yaml, llm.yaml, llm_zeroshot.yaml      model recipes
│   ├── data.yaml, tasks.yaml, lexicon.yaml, corpus.yaml, dapt/
│   ├── results.yaml                                    which runs make up the tables
│   └── kaggle/                                         our Kaggle session configs (lanes, pilot)
├── src/matcha_bench/
│   ├── data/ tasks/          MATCHA loading, cleaning, user split; task builders
│   ├── lexicon/ pretrain/    lexicon mining and filtering; DAPT corpus, guided-masking collator, trainer
│   ├── finetune/ baselines/  BERT-type fine-tuning and metrics; TF-IDF + LR
│   ├── llm/                  MentalGLM checkpoint conversion, prompts, QLoRA training, inference
│   └── results.py stats.py   results tables, paired bootstrap
├── scripts/                  one CLI per step, run_pipeline.py, make_tables.py; kaggle/ = our tooling
├── notebooks/                the reproduction notebooks 01–06
├── tests/                    offline unit tests
└── docs/                     BASELINES.md (recipes, deviations), LEXICON.md, KAGGLE.md
```

## Ethics and data use

- MATCHA contains posts about people's mental health. It is distributed under a DUA for
  non-commercial research. **No data, derived post-level files, user IDs or lexicons are included
  here.**
- Only aggregated, k-anonymous candidate **words** were sent to external LLMs (lexicon filtering).
  Posts never left the machine, which is also why the LLM baseline answers with labels only.
- Models trained here detect **community membership and self-disclosure on social media**. They do not
  diagnose anyone and must not be used to screen or label individuals.
- Known biases: disorder cohorts are about 82% female vs 62% of controls; labels for identified-positive
  users come from an LLM and are noisy; ADHD and ASD posts are often written by caregivers.

## Citation

```bibtex
@misc{matchabench2026,
  title  = {MATCHA-Bench: Baselines for Multi-Disorder Mental Health Detection on Chinese Social Media},
  year   = {2026},
  note   = {Code: https://github.com/shehryars715/matcha-bench}
}
```

Please also cite the work this builds on: MATCHA (Li et al., 2025), Chinese MentalBERT (Zhai et
al., 2024), MentalGLM (Zhai et al., 2025), Murarka et al. (2021), the Chinese BERT/RoBERTa models (Cui et
al., 2021), the depression lexicon (Li et al., 2020), and the evaluation datasets (SMP2020-EWECT;
Qi et al., 2023; SOS-HL-1K; the RoBERTa paper's Reddit data).

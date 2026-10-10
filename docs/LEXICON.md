# The multi-disorder lexicon

How the lexicon behind `matcha-multidisorder` was built. The word list itself is derived from
DUA-restricted data and is not tracked in this repository (see `lexicons/README.md`).

## Why a new lexicon

Chinese MentalBERT masks words from a depression lexicon (Li et al., 2020) before topping up with
random words. On MATCHA that lexicon fills only **10.3%** of the masking budget, so "guided"
masking is mostly random. MATCHA covers seven disorders, so the lexicon should too.

## Pipeline

| Step | Method | Output |
|---|---|---|
| 1. Data | Training-split users only (29,222 users, ~1.0M posts); dev/test users never influence the lexicon | `data/processed` |
| 2. Candidate mining | jieba segmentation; per disorder vs clean controls, weighted log-odds with an informative Dirichlet prior (Monroe et al., 2008); z ≥ 3, ≥ 5 distinct users (k-anonymity), top 1,500 per disorder | 9,307 (disorder, word) pairs, 4,269 unique words (`scripts/mine_lexicon.py`) |
| 3. Annotator A | LLM, single pass over the 4,269 unique words with the criteria below | 526 words (`lexicons/llm_annotations.csv`) |
| 4. Annotator B | DeepSeek (`deepseek-chat`), each (disorder, word) judged 3× at temperature 0.7, majority vote | 1,439 words; 97.1% vote self-consistency (`scripts/filter_lexicon_deepseek.py`) |
| 5. Consensus | Keep a word only if both annotators keep it | **520 words** (`scripts/build_multidisorder_lexicon.py`) |

**Criteria.** Keep a word only if it is clearly related to mental health or to the lived
experience of a disorder: emotion, cognition, somatic, behaviour, clinical, community slang, or
social stressor. Reject function words, everyday topics, named entities, advertising, platform
boilerplate, and physical illness unrelated to mental health.

**Privacy.** Only candidate words (with the disorder community name) are sent to an LLM, never
posts, user IDs or timestamps.

## Validation without human annotators

| Measure | Value |
|---|---|
| Word level (n = 4,269): agreement / Cohen's κ | 78.3% / 0.43 |
| (Disorder, word) level (n = 9,307): agreement / Cohen's κ | 82.4% / 0.56 |
| Annotator A's words also kept by B | 520 / 526 (98.9%) |
| Same category among shared words | 86.5% |

κ is moderate mainly because annotator B is far more inclusive. The two annotators directly
contradict each other on only 6 words. Reproduce with `scripts/compare_lexicon_annotations.py`.

## Composition and coverage

By category: emotion 129, clinical 132, cognition 95, behaviour 72, somatic 58, social 24,
community 10.

| Lexicon | Words | Chunks with a lexicon word | Share of masked tokens from the lexicon |
|---|---|---|---|
| Depression (Li et al., 2020) | 1,470 | 53.4% | 10.3% |
| **Multi-disorder consensus** | **520** | **62.8%** | **17.3%** |
| Annotator B alone | 1,439 | 90.5% | 39.8% |

These figures come from 10k chunks sampled from the full DAPT corpus (281k chunks). The more
inclusive annotator-B lexicon is available as an optional ablation (`lexicons/lexicon_deepseek.txt`).
To use it, add it to `userdict_lexicons` in `configs/corpus.yaml` and rebuild the corpus cache, so
that segmentation stays identical across variants.

## Limitations

- No human annotation; validity rests on agreement between two independent LLMs.
- Words are judged out of context. Ambiguous words (e.g. 双向, a frequent misspelling of 双相
  "bipolar") are judged by their dominant use in these communities.
- The control group is keyword-sampled and not clinically screened.

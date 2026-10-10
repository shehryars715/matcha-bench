# Lexicons

Lexicon files are generated locally and are not tracked by git. The depression lexicon belongs
to its original authors, and the mined lexicon is derived from DUA-restricted data, so check
the MATCHA DUA before releasing the word lists themselves.

| File | Built by | Description |
|---|---|---|
| `lexicon_depression.txt` | `scripts/build_depression_lexicon.py` | Chinese depression-domain lexicon (Li et al., 2020), filtered exactly as in the Chinese MentalBERT code (抑郁倾向 = 1, Chinese characters only). 1,470 words. |
| `candidates.csv` | `scripts/mine_lexicon.py` | Per-disorder words over-represented relative to clean controls (log-odds with informative Dirichlet prior; z ≥ 3; used by ≥ 5 distinct users). |
| `llm_annotations.csv` | LLM annotator (single pass over the 4,269 unique candidates) | `word,category` for every word judged mental-health related; anything not listed is rejected. |
| `multidisorder_annotations.csv` | `scripts/build_multidisorder_lexicon.py` | Every (disorder, candidate) row with its keep decision and category. |
| `lexicon_multidisorder.txt` | same | **The multi-disorder lexicon used for masking** (526 words; union across the 7 disorders). |
| `deepseek_*.{jsonl,csv}`, `lexicon_deepseek.txt` | `scripts/filter_lexicon_deepseek.py` | Optional alternative filter: DeepSeek, 3 votes per word, cached. |

Format: one word per line, UTF-8, `#` starts a comment line.
Categories: `emotion`, `cognition`, `somatic`, `behavior`, `clinical`, `community`, `social`.

## Filtering rule

Keep a word only if it is clearly related to mental health or to the lived experience of a
disorder: symptoms, emotions, thought patterns, behaviours, diagnosis and treatment terms,
community slang, or closely tied relational stressors. Reject function words, everyday topics,
named entities (people, brands, places, games), advertising, platform boilerplate, and general
physical illness unrelated to mental health.

## What is sent to an external LLM

Only candidate **words**, with the name of the disorder community they were mined from. No
posts, user IDs or timestamps are ever sent. Every word must have been written by at least
`min_users` distinct users (k-anonymity), so rare strings that could identify a person are
never sent.

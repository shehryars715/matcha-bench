# Lexicons

Lexicon files are generated locally and are not tracked by git. The depression lexicon belongs
to its original authors, and the mined lexicon is derived from DUA-restricted data, so check
the MATCHA DUA before releasing the word lists themselves.

| File | Built by | Description |
|---|---|---|
| `lexicon_depression.txt` | `scripts/build_depression_lexicon.py` | Chinese depression-domain lexicon (Li et al., 2020), filtered exactly as in the Chinese MentalBERT code (抑郁倾向 = 1, Chinese characters only). About 1.5k words. |
| `candidates.csv` | `scripts/mine_lexicon.py` | Per-disorder words over-represented relative to clean controls (log-odds with informative Dirichlet prior; z ≥ 3; used by ≥ 5 distinct users). |
| `deepseek_cache.jsonl` | `scripts/filter_lexicon_deepseek.py` | Raw judgements, one per (model, prompt version, disorder, word, vote). Re-runs are free. |
| `deepseek_annotations.csv` | same | Majority vote, category and vote agreement for every candidate. |
| `lexicon_deepseek.txt` | same | The multi-disorder lexicon used for masking: union of kept words across the 7 disorders. |

Format: one word per line, UTF-8, `#` starts a comment line.

## What is sent to DeepSeek

Only candidate **words**, with the name of the disorder community they were mined from. No
posts, user IDs or timestamps are ever sent. Every word must have been written by at least
`min_users` distinct users (k-anonymity), so rare strings that could identify a person are
never sent.

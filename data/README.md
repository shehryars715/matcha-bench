# Data

Nothing in this folder except this README is tracked by git. MATCHA is released under a
**Data Use Agreement** (gated access, CC BY-NC-SA 4.0), so neither the raw data nor anything
derived from it at post level may be committed or shared publicly.

## 1. MATCHA (pre-training corpus and MATCHA tasks)

Request access from the dataset authors: sign the DUA from the
[MATCHA repository](https://github.com/Baihan-12/MATCHA-Mental-Health-Dataset/) and request
access on ModelScope. Then either

* unpack it next to the repo root as `MATCHA_Multi-Disorder_mental_health_daTa_in_CHinA/`, or
* point to it with `export MATCHA_ROOT=/path/to/MATCHA` (or `matcha_root:` in `configs/data.yaml`).

Expected layout:

```
MATCHA_ROOT/
├── Homepage_Posts/homepage_posts_{ADHD,ASD,BD,DEP,GAD,OCD,SZ,CONTROL}.csv
├── Super_Topic_Posts/super_topic_posts_{ADHD,ASD,BD,DEP,GAD,OCD,SZ}.csv
└── Identified_Positive_User/positive_users_{ADHD,ASD,BD,DEP,GAD,OCD,SZ}.csv
```

> The GitHub mirror of MATCHA omits `homepage_posts_DEP.csv` (163 MB). Use the ModelScope copy.

## 2. Downstream datasets from Chinese MentalBERT

```bash
python scripts/download_mentalbert_data.py   # EWECT, Waimai_10k, SocialCD-3K, SOS-HL-1K -> data/raw/mentalbert/
```

C2D2 is shared on request by its authors. Save it as `data/raw/mentalbert/c2d2/{train,dev,test}.csv`
with columns `text,label`, and it is picked up automatically.

## 3. Generated folders

| Path | Produced by | Contents |
|---|---|---|
| `data/processed/` | `prepare_matcha.py` | cleaned posts, users, IP anchors, global user-level split |
| `data/tasks/<task>/` | `build_tasks.py` | `train/dev/test.jsonl` + `meta.json` for every task |
| `data/dapt/` | `build_dapt_corpus.py` | packed 128-token chunks with word-boundary metadata |
| `data/raw/mentalbert/` | `download_mentalbert_data.py` | public downstream datasets |

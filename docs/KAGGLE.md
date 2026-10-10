# Running on Kaggle

The GPU work (DAPT and fine-tuning) runs on Kaggle's free accelerators. CPU-heavy preparation
runs best locally.

## 0. Unlock the GPU

Kaggle → *Settings* → *Phone verification*. GPUs (T4 ×2 or P100) become available with about
30 h/week and 12 h per session. Choose **T4 ×2**: fp16 is fast on T4, and the trainer uses both
GPUs automatically.

## 1. Prepare data locally (CPU, ~20 min)

```bash
python scripts/prepare_matcha.py
python scripts/download_acl_data.py
python scripts/build_tasks.py
python scripts/build_depression_lexicon.py
python scripts/mine_lexicon.py
python scripts/filter_lexicon_deepseek.py      # needs DEEPSEEK_API_KEY, < $1
python scripts/build_dapt_corpus.py
```

Then zip `data/processed`, `data/tasks`, `data/dapt` and `lexicons/`, and upload them as a
**private** Kaggle dataset, e.g. `matcha-bench-data`.

> Check the MATCHA DUA before uploading derived data to a third-party platform. Keep the
> dataset private, and never make it public.

## 2. Get the code into the notebook

Either
- push this repo to a private GitHub repo and `git clone` it with a token stored in Kaggle
  *Secrets*, or
- upload the repo folder as a second private Kaggle dataset.

## 3. Run

Open `notebooks/kaggle_matcha_bench.ipynb` on Kaggle, attach both datasets, choose GPU T4 ×2,
and run all cells. For long jobs use **Save Version → Save & Run All** so the job runs headless
for up to 12 h, and `/kaggle/working` (≤ 20 GB) is kept as output.

## 4. Resuming across sessions

- **DAPT** saves a checkpoint every `save_steps` and resumes automatically from the last one in
  `output_dir`. To continue in a new session, attach the previous notebook's output as an input
  and copy `outputs/dapt/<variant>` back into `/kaggle/working/matcha-bench/outputs/dapt/`.
- **Fine-tuning** skips any `(task, model, seed)` that already has a `metrics.json`. Copy the
  previous `outputs/finetune` back the same way and re-run.

## 5. Suggested session plan

| Session | Work |
|---|---|
| 1 | smoke test, then `random` DAPT |
| 2 | `depression` DAPT, then `deepseek` DAPT |
| 3 | ACL tasks × all models × 3 seeds |
| 4–5 | MATCHA tasks × all models × 3 seeds |
| local | `run_tfidf.py`, `summarize_results.py` |

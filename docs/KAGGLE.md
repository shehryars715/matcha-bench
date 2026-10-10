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
python scripts/filter_lexicon_deepseek.py      # annotator B, key in .env (~$0.26)
python scripts/build_multidisorder_lexicon.py  # consensus with lexicons/llm_annotations.csv
python scripts/build_dapt_corpus.py
```

Then build the upload zip (only what the GPU stage needs: the DAPT corpus, the tasks and the
lexicon word lists; raw posts, candidates and annotations stay local):

```bash
python scripts/pack_kaggle_data.py        # -> ../matcha-bench-data.zip
python scripts/kaggle_job.py upload-data  # or upload the zip in the Kaggle web UI
```

Upload it as a **private** Kaggle dataset named `matcha-bench-data` (Datasets → New Dataset).

> Check the MATCHA DUA before uploading derived data to a third-party platform. Keep the
> dataset private, and never make it public.

## 2. Get the code into the notebook

The notebook clones the GitHub repo. A public repo needs no setup. For a private repo:

1. GitHub → Settings → Developer settings → *Fine-grained tokens* → Generate. Give it access to
   `matcha-bench` only, with permission **Contents: Read-only**.
2. In the Kaggle notebook: *Add-ons → Secrets → Add*, label `GITHUB_TOKEN`, paste the token, and
   tick it for this notebook.

The notebook clones the repo with the token, then removes the token from the clone's git config,
because `/kaggle/working` is saved as notebook output.

## 3. Run

**From the command line (recommended).** `scripts/kaggle_job.py` pushes each step as a headless GPU
job (T4 ×2, Internet on) and downloads its outputs:

```bash
python scripts/kaggle_job.py push --name smoke --cmd "python scripts/run_dapt.py --config configs/dapt/smoke.yaml"
python scripts/kaggle_job.py status --name smoke
python scripts/kaggle_job.py output --name smoke
```

Jobs listed in `--inputs-from` have their `outputs/` merged in, so DAPT resumes and fine-tuning
finds the trained models.

**Or in the browser:**

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
| 2 | `depression` DAPT, then `multidisorder` DAPT |
| 3 | ACL tasks × all models × 3 seeds |
| 4–5 | MATCHA tasks × all models × 3 seeds |
| local | `run_tfidf.py`, `summarize_results.py` |

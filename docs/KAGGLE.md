# Running on Kaggle

The GPU work (DAPT and fine-tuning) runs on Kaggle's free accelerators. CPU-heavy preparation
runs best locally.

## 0. Unlock the GPU

Kaggle → *Settings* → *Phone verification*. GPUs (T4 ×2 or P100) become available with about
30 h/week and 12 h per session. Choose **T4 ×2**: fp16 is fast on T4, and the trainer uses both
GPUs automatically (DataParallel). For fine-tuning this means a per-GPU batch of 16 gives an
effective batch of 32.

To check the weekly quota, run `python -m kaggle quota`. In Kaggle CLI 2.2.3 it can crash with
`not enough values to unpack`; the usage is also shown on the Kaggle website under *Settings*.

## 1. Prepare data locally (CPU, ~20 min)

```bash
python scripts/prepare_matcha.py
python scripts/download_mentalbert_data.py
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

## 3. Run the whole pipeline in resumable sessions (recommended)

**In the browser:** import `notebooks/reproduce_kaggle.ipynb` into Kaggle, attach the
dataset, set `MODE = "full"` and *Save & Run All*. To continue later, attach the previous version's
output and run it again.

**From the command line:**

```bash
python scripts/kaggle_pipeline.py start      # start the next session (hard cap: budget_hours, default 9 h)
python scripts/kaggle_pipeline.py status     # queued / running / complete
python scripts/kaggle_pipeline.py log        # what the last session did
python scripts/kaggle_pipeline.py results    # download result tables and metrics
```

Each `start` creates a new Kaggle job (`matcha-bench-run-001`, `-002`, ...), attaches the previous
session's output and runs `scripts/run_pipeline.py`. That script works through
`configs/pipeline.yaml`:
1. preflight check;
2. DAPT random → DAPT depression → DAPT multi-disorder;
3. fine-tuning on the Chinese MentalBERT tasks (sanity check, 6 models × 3 seeds);
4. fine-tuning on the MATCHA tasks (the baselines, 5 models × 3 seeds);
5. TF-IDF, then summary tables and the model comparisons.

It skips anything already finished and resumes the current DAPT run from its last checkpoint. It
also stops 30 minutes before the cap, saving a checkpoint, so the session ends cleanly and its
outputs are kept. Just run `start` again once a session has finished, until the log says
`pipeline complete`.

Kaggle clones the code from GitHub at the start of each session, so changes to the pipeline or
configs must be pushed first. Don't start a session that doesn't fit in the remaining weekly
quota: running out of quota stops the session abruptly, without the clean checkpoint stop.

## 3b. Run single steps

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

## 5. Session plan (measured)

Each 9-hour session does about 8.5 hours of work. DAPT takes about 3.6 h per variant.

| Session | Work |
|---|---|
| 1 | preflight, `random` DAPT, `depression` DAPT, `multidisorder` DAPT (to step 2,290 of 6,594) |
| 2 | `multidisorder` DAPT finished; Suicide, Cognitive and Waimai fine-tuning |
| 3 | Waimai finished, EWECT-epidemic, EWECT-usual (7 of 18 runs) |
| 4–10 | rest of EWECT-usual, then the MATCHA tasks (≈ 55 GPU-h, estimated) |
| end | TF-IDF and summary (CPU, inside the last session) |

The full plan is about 80 GPU-h in total; see `docs/RESEARCH_PLAN.md` §5 for the breakdown and the
options to cut it.

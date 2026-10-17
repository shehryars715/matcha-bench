# Running on Kaggle

The GPU work (DAPT and fine-tuning) runs on Kaggle's free accelerators. CPU-heavy preparation
runs best locally.

## 0. Unlock the GPU

Kaggle → *Settings* → *Phone verification*. GPUs (T4 ×2 or P100) become available with about
30 h/week and 12 h per session. Choose **T4 ×2**: fp16 is fast on T4, and the trainer uses both
GPUs. DAPT splits each batch across both GPUs (DataParallel). Fine-tuning runs one
(task, model, seed) per GPU at the same time (`finetune_one_run_per_gpu` in
`configs/pipeline.yaml`), each with a true batch of 16; per-run output goes to
`outputs/logs/finetune/`. With that switch off, one run uses both GPUs and its effective batch is
32. The sanity-check tasks were run that way.

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
4. fine-tuning on the 4 core MATCHA tasks (the baselines, 4 models × 3 seeds, one run per GPU);
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

## 3c. The LLM baseline (MentalGLM)

The LLM baseline has its own pipeline file, run in the same kind of sessions:

```bash
python scripts/kaggle_pipeline.py start --pipeline configs/pipeline_llm.yaml
```

- **Model.** The released `zwzzz/MentalGLM-chat` (GLM-4-9B) is downloaded and converted to the
  native transformers `glm` format at the start of each session (about 19 GB, kept in `/tmp`, not
  in the output). The released checkpoint ships code for transformers 4.43, which no longer runs on
  current versions. `src/matcha_bench/llm/convert.py` explains the conversion; it reproduces the
  original model's outputs exactly (checked on small random models).
- **Sanity check** (`llm_sanity` stage). MentalGLM-chat labels its own test sets, so the scores
  compare with the paper's 85.12 (SOS-HL-1K) and 71.04 (SocialCD-3K). It runs twice: in fp16
  across both GPUs, and in 4-bit on one GPU, which is the setting the MATCHA fine-tuning uses.
- **Timing probes** (`llm_probe` stage). A few LoRA steps on the shortest and the longest MATCHA
  inputs. The measured speeds set the training sizes in `configs/llm.yaml`.
- **MATCHA fine-tuning** (`finetune-llm` stage, switched on after the probes). One 4-bit LoRA run
  per GPU and task, with the paper's settings. See `configs/llm.yaml` for the full list.

Results go to `outputs/llm/` (`sanity/`, `finetune/<task>/<model>/seed<k>/`, `finetune/_probe/`).

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
| 4–5 | the 4 core MATCHA tasks, two runs at a time (≈ 16 h, estimated; EWECT-usual stays at 7 of 18 runs) |
| end | TF-IDF and summary (CPU, inside the last session) |

The full plan is about 41 hours of session time; see `docs/RESEARCH_PLAN.md` §5 for the breakdown and
the runs that were cut.

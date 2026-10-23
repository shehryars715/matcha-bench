# How the reported runs were made (Kaggle)

Every GPU run behind the reported numbers ran on Kaggle's free accelerators (T4 × 2, 16 GB each;
30 GPU-hours per week per account; sessions capped at 9 h here). The CPU steps ran on a laptop.
Nothing on this page is needed to reproduce the results: `scripts/run_pipeline.py --no-limit` with
the pipelines in `configs/` does the same work on any GPU (see the README). This is the tooling that
split the work into sessions.

| Path | What |
|---|---|
| `scripts/kaggle/pack_kaggle_data.py` | packs what the GPU stage needs (DAPT corpus, tasks, lexicon word lists) into one zip |
| `scripts/kaggle/kaggle_job.py` | uploads that zip as a **private** Kaggle dataset; pushes single steps as headless GPU jobs |
| `scripts/kaggle/kaggle_pipeline.py` | chains sessions: each one attaches the previous session's output, restores `outputs/`, and runs `scripts/run_pipeline.py` with a deadline |
| `configs/kaggle/` | the session configs that split the MentalGLM runs over accounts (lanes) and the capped pilot |
| `outputs/kaggle/` (not tracked) | session state and the downloaded outputs |

## Sessions

```bash
python scripts/kaggle/pack_kaggle_data.py                    # -> ../matcha-bench-data.zip
python scripts/kaggle/kaggle_job.py upload-data              # private dataset matcha-bench-data
python scripts/kaggle/kaggle_pipeline.py start [--lane N] [--pipeline CFG] [--budget-hours H] [--margin-minutes M]
python scripts/kaggle/kaggle_pipeline.py status | log | results [--lane N]
python scripts/kaggle/kaggle_pipeline.py account --token-file FILE --data-owner USER [--lane N]
```

- A session clones the code from GitHub, so changes must be pushed before `start`.
- `run_pipeline.py` stops `safety_margin_minutes` before the cap and saves; the next `start` resumes
  from the last completed session (DAPT from its last checkpoint, LLM runs from their last epoch,
  finished runs skipped). Without `--pipeline`, `start` reuses the resumed session's pipeline.
- Two runs share a session (one per T4), and the quota counts session time, so each quota hour
  does two runs' work. Running out of quota mid-session kills it without the clean stop.
- A **lane** (`--lane N`) is an independent chain of sessions with its own state file
  (`outputs/kaggle/pipeline_state_<N>.json`), usually on another account.

> The MATCHA DUA applies to the derived data in the zip: keep the Kaggle dataset private.

## What ran where

| Work | Pipeline | Sessions |
|---|---|---|
| Chinese MentalBERT: 3 DAPT variants, sanity tasks, MATCHA tasks, TF-IDF | `configs/pipeline_mentalbert.yaml` (then also holding the multi-disorder stages) | run-001 – run-004 |
| MentalGLM: sanity check, timing probes, capped runs (≤ 4,000 training examples, 500-item dev sample; the pilot/ablation) and the full-data pilot | `configs/kaggle/pipeline_llm.yaml` | run-005 – run-010 |
| MentalGLM final runs (`mentalglm-chat-full`) | `configs/kaggle/pipeline_llm_full_{1,2,3}.yaml` | lanes 1–3 |
| MentalGLM zero-shot rows | `configs/kaggle/pipeline_llm_zeroshot.yaml` | lane 1 |
| RoBERTa on MATCHA, Reddit sanity check | `configs/pipeline_roberta.yaml` | lane `roberta` |

`configs/pipeline_mentalglm.yaml` gathers the MentalGLM runs of the lane configs (and the binary
task's `mentalglm-chat` runs) into one pipeline.

## Measured session time

| Work | Session time |
|---|---|
| DAPT, 3 variants (2 GPUs, DataParallel) | 10.8 h |
| Chinese MentalBERT sanity tasks (6 models × 4 tasks × 3 seeds + 7 EWECT-usual runs) | ≈ 14 h |
| Chinese MentalBERT MATCHA tasks (48 runs, 2 at a time) + TF-IDF + summary | ≈ 6 h |
| MentalGLM final runs (12 runs, 2 at a time) | ≈ 50 h (≈ 100 GPU-h) |
| MentalGLM zero-shot rows | ≈ 2.7 h |
| RoBERTa (12 runs + Reddit) | 3.8 h |

## Gotchas

- `python -m kaggle quota` crashes in CLI 2.2.3 (it cannot parse the quota's timedelta). The API
  works once `kagglesdk.kaggle_object.TimeDeltaSerializer._from_dict_value` is patched to parse
  `"<seconds>s"`; read `gpu_quota.time_used` as a timedelta (printing it as JSON drops whole days).
- A session can stay "running" with an empty log and no Docker image, then error 1–3.5 h later
  without using quota: start it again.
- `results` skips `test_predictions.npz`; fetch them with
  `python -m kaggle kernels output <owner>/<slug> -p outputs/kaggle/<slug> --file-pattern "test_predictions"`.
- The MentalGLM checkpoint ships code for transformers 4.43, which no longer runs; it is converted to
  the native `glm` format at the start of each session (~3.5 min, kept in `/tmp`).

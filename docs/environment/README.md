# Software environment of the reported runs

## Exact: the Kaggle Docker images

Every GPU session's image, read from its kernel metadata (`docker_image`):

| Image (Kaggle GPU image) | Sessions |
|---|---|
| `gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f` | every session up to 2026-10-22: all Chinese MentalBERT, RoBERTa and zero-shot runs and all MentalGLM runs except the two below |
| `gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435` (Kaggle's image release of 2026-10-22, a certificate patch with no library changes) | the last MentalGLM sessions: `disorder_post_masked` seed 43 (epoch 2 and test) and `disorder_post` seed 45 |

Each MentalGLM session also ran `pip install "peft>=0.13" "bitsandbytes>=0.45" "transformers>=4.46"`,
and the RoBERTa session `pip install "gdown>=5.2"`, so those packages were the newest releases on the
day of the session (transformers was already new enough and was not changed).

## Approximate: library versions

The GPU images' package lists were not recorded during the runs. `kaggle_cpu_image_2026-10-22_pip_freeze.txt`
is the full `pip freeze` of Kaggle's **CPU** image built on 2026-10-22 (the same day as the second GPU
image), after the same `pip install` lines, captured on 2026-10-24. Kaggle builds both images from
the same package set, but the CPU image has CPU-only PyTorch, and peft and bitsandbytes were
installed on 2026-10-24, so treat it as close rather than exact. Key versions:

| Package | Version |
|---|---|
| Python | 3.13.15 |
| torch | 2.11.0 (CPU build here; CUDA build in the GPU image) |
| transformers | 5.16.1 |
| tokenizers | 0.23.1 |
| huggingface_hub | 1.29.0 |
| accelerate | 1.14.0 |
| peft | 0.20.0 |
| bitsandbytes | 0.50.2 |
| safetensors | 0.8.0 |
| numpy | 2.1.3 |
| pandas | 2.3.3 |
| scikit-learn | 1.6.1 |
| scipy | 1.16.3 |
| pyarrow | 23.0.1 |
| jieba | 0.42.1 |
| gdown | 5.2.2 |

The CPU steps (data preparation, tasks, lexicons, DAPT corpus, TF-IDF, tables) ran on Windows 11
with Python 3.14 and transformers 5.9. The code supports transformers 4.41 (4.46 for MentalGLM) up
to 5.x.

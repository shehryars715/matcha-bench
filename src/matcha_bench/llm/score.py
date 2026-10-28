"""Label probabilities from a fine-tuned LLM: one forward pass per item, no generation.

A fine-tuned answer opens with a fixed sentence that names the label ("\\n这篇帖子来自抑郁症社区。").
Every label's answer is tokenised in full, exactly as in training; the tokens they all share
("\\n这篇帖子来自") form the answer prefix, and the next token is different for every label (checked;
with the GLM-4 tokenizer: 注意 孤独 双 抑郁症 广泛 强迫 精神, and 对照组 for the control group). The
prompt plus that prefix goes through the model once, and the next-token distribution gives each
label's probability of being the one the answer names. Greedy generation names the label whose
token is most probable, so the arg-max reproduces the run's generated test answers (checked on
every run before the dev and training sets are scored).

What the probabilities are for: correcting for the difference between the capped, near-balanced
training sets and the natural class mix of dev and test (a prior shift), and comparing models on
more than their hard labels. Output per split: <split>_scores.npz with `logp` (n x labels, the log-
probability of each label's token over the whole vocabulary), `labels` and `indices` (positions in
the split file). Long splits are saved in chunks, so a session cap loses at most one chunk.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from ..finetune.metrics import classification_metrics
from ..tasks import load_task
from ..utils import get_logger, save_json
from .infer import build_prompt_ids, load_model
from .prompts import TaskPrompt, get_prompt

log = get_logger()


def answer_tokens(tokenizer, prompt: TaskPrompt) -> tuple[list[int], list[int]]:
    """(token ids every label's answer starts with, the next token of each label's answer)."""
    if prompt.multi_label:
        raise ValueError("label probabilities need one label per item")
    answers = [tokenizer.encode("\n" + prompt.target(i), add_special_tokens=False)  # as in training
               for i in range(len(prompt.verbalizers))]
    n = 0
    while all(len(a) > n for a in answers) and len({a[n] for a in answers}) == 1:
        n += 1
    if any(len(a) == n for a in answers):
        raise ValueError("one label's answer is the start of another's")
    first = [a[n] for a in answers]
    if len(set(first)) < len(first):
        raise ValueError(f"labels share their first token after the common prefix: {prompt.verbalizers}")
    return answers[0][:n], first


@torch.no_grad()
def label_logprobs(model, sequences: list[list[int]], label_ids: list[int], pad_id: int, batch_size: int = 16,
                   deadline: float | None = None) -> np.ndarray:
    """log P(next token = label_ids[j] | sequence) for every sequence, as an (n, labels) float32 array
    in input order. Batches are left-padded, longest first, with the position ids generation uses;
    only the last position is projected to the vocabulary."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    device = next(model.parameters()).device
    cuda = device.type == "cuda"
    cols = torch.tensor(label_ids, device=device)
    out = np.zeros((len(sequences), len(label_ids)), dtype=np.float32)
    order = sorted(range(len(sequences)), key=lambda i: -len(sequences[i]))

    def run(idx):
        width = max(len(sequences[i]) for i in idx)
        ids = torch.full((len(idx), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(idx), width), dtype=torch.long)
        for row, i in enumerate(idx):
            ids[row, width - len(sequences[i]):] = torch.tensor(sequences[i])
            mask[row, width - len(sequences[i]):] = 1
        positions = (mask.cumsum(-1) - 1).clamp(min=0)
        with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):  # as in generation
            hidden = base.model(input_ids=ids.to(device), attention_mask=mask.to(device),
                                position_ids=positions.to(device), use_cache=False).last_hidden_state[:, -1]
            logits = base.lm_head(hidden)
        logp = torch.log_softmax(logits.float(), dim=-1)[:, cols]
        out[idx] = logp.cpu().numpy()

    pending = [order[s:s + batch_size] for s in range(0, len(order), batch_size)]
    while pending:
        if deadline and time.time() > deadline:
            raise SystemExit("deadline reached while scoring; rerun next session")
        idx = pending.pop(0)
        try:
            run(idx)
        except torch.OutOfMemoryError:
            if len(idx) == 1:
                raise
            torch.cuda.empty_cache()
            log.warning("out of GPU memory on a batch of %d; retrying in halves", len(idx))
            pending[:0] = [idx[:len(idx) // 2], idx[len(idx) // 2:]]
    return out


def _score_split(model, tokenizer, prompt, texts, prefix, label_ids, hp, partial: Path, deadline) -> np.ndarray:
    """Scores for one split, resumed from (and saved to) `partial` every `save_every` items."""
    seqs = [build_prompt_ids(tokenizer, prompt, t, hp.get("max_text_tokens"), hp.get("max_text_chars")) + prefix
            for t in texts]
    done = np.load(partial)["logp"] if partial.exists() else np.zeros((0, len(label_ids)), np.float32)
    if len(done):
        log.info("resuming %s at item %d of %d", partial.name, len(done), len(seqs))
    chunk = int(hp.get("save_every", 2048))
    while len(done) < len(seqs):
        part = label_logprobs(model, seqs[len(done):len(done) + chunk], label_ids, tokenizer.pad_token_id,
                              int(hp.get("score_batch_size", 16)), deadline=deadline)
        done = np.concatenate([done, part])
        np.savez_compressed(partial, logp=done)
        log.info("%s: %d of %d scored", partial.name, len(done), len(seqs))
    return done


def _split_stats(logp: np.ndarray, labels: np.ndarray, meta: dict) -> dict:
    preds = logp.argmax(1)
    m = classification_metrics(labels, preds, meta["problem_type"], meta["average"], meta["label_names"],
                               labels_list=list(range(len(meta["label_names"]))))
    mass = np.exp(logp).sum(1)  # probability the answer names some label at all
    return {"f1": m["f1"], "accuracy": m["accuracy"], "per_class_f1": m.get("per_class_f1"), "n": len(labels),
            "label_mass_mean": round(float(mass.mean()), 6), "label_mass_min": round(float(mass.min()), 6)}


def run_llm_scores(task_root: Path, task: str, model_path: str, model_tag: str, adapter_root: Path,
                   output_root: Path, seed: int, hp: dict, splits=("test", "dev", "train"),
                   deadline: float | None = None) -> dict | None:
    """Label probabilities of one fine-tuned run (its best_adapter) on each split, test first. The
    test arg-max must agree with the run's generated test answers on at least `min_agreement` of the
    items, or the run stops before scoring the other splits (check_failed.json)."""
    out_dir = Path(output_root) / task / model_tag / f"seed{seed}"
    if (out_dir / "metrics.json").exists():
        log.info("Skipping %s / %s / seed %d scores (already done)", task, model_tag, seed)
        return None
    run_dir = Path(adapter_root) / task / model_tag / f"seed{seed}"
    adapter = run_dir / "best_adapter"
    for needed in (adapter / "adapter_model.safetensors", run_dir / "metrics.json", run_dir / "test_predictions.npz"):
        if not needed.exists():
            raise FileNotFoundError(f"{needed} missing: the fine-tuned run must be finished (and its outputs restored)")
    t_start = time.time()
    meta, data = load_task(task_root, task)
    prompt = get_prompt(task)
    model, tokenizer = load_model(model_path, hp["quant"], lora_path=str(adapter))
    prefix, label_ids = answer_tokens(tokenizer, prompt)
    out_dir.mkdir(parents=True, exist_ok=True)
    limit = hp.get("limit")
    record = {"task": task, "model_tag": model_tag, "seed": seed, "model_path": str(model_path),
              "adapter": str(adapter), "label_names": meta["label_names"], "hyperparameters": hp,
              "answer_prefix": tokenizer.decode(prefix), "label_tokens": [tokenizer.decode([t]) for t in label_ids],
              "splits": {}}
    generated = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["test"]
    for split in splits:
        df = data[split].reset_index(drop=True)
        if limit:
            df = df.head(int(limit))
        labels = df["label"].astype(int).to_numpy()
        path, partial = out_dir / f"{split}_scores.npz", out_dir / f"{split}_scores.partial.npz"
        if path.exists():
            z = np.load(path)
            logp, seconds = z["logp"], float(z["seconds"])
        else:
            t0 = time.time()
            logp = _score_split(model, tokenizer, prompt, df["text"].tolist(), prefix, label_ids, hp, partial,
                                deadline)
            seconds = time.time() - t0  # this session's part only, if the split was resumed
            np.savez_compressed(path, logp=logp, labels=labels, indices=np.arange(len(df)), seconds=seconds)
            partial.unlink(missing_ok=True)
        stats = _split_stats(logp, labels, meta)
        stats.update({"seconds": round(seconds, 1), "items_per_second": round(len(df) / max(seconds, 1e-9), 2)})
        if split == "test":
            z = np.load(run_dir / "test_predictions.npz")
            gen = z["preds"][np.argsort(z["indices"])][:len(df)]
            stats["agreement_with_generated"] = round(float((logp.argmax(1) == gen).mean()), 6)
            stats["generated_f1"] = generated["f1"] if not limit else None
            if stats["agreement_with_generated"] < float(hp.get("min_agreement", 0.98)):
                save_json(stats, out_dir / "check_failed.json")
                raise RuntimeError(f"{task}/{model_tag}/seed{seed}: label scores agree with the generated test "
                                   f"answers on only {stats['agreement_with_generated']:.4f}; see check_failed.json")
        record["splits"][split] = stats
        log.info("%s | %s | seed %d | %s: F1 %.4f, %.1f items/s%s", task, model_tag, seed, split, stats["f1"],
                 stats["items_per_second"],
                 f", agreement {stats['agreement_with_generated']:.4f}" if split == "test" else "")
    record["minutes_total"] = round((time.time() - t_start) / 60, 1)
    save_json(record, out_dir / "metrics.json")
    return record

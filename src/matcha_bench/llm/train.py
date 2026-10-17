"""LoRA fine-tuning of a (converted) MentalGLM model on one task, label-only answers (QLoRA on T4).

Protocol, following MentalGLM's stage-2 training (LLaMA-Factory, LoRA) as far as a T4 allows:
* the released model is the starting point; LoRA adapters (rank 8, alpha 16, LLaMA-Factory's
  defaults) on every linear layer; the base model is frozen in 4-bit NF4 (QLoRA);
* AdamW, peak lr 1e-4, 1% warm-up then linear decay, effective batch 32 (4 x 8 accumulation),
  fp16, loss on the answer tokens only (LLaMA-Factory's default), inputs capped at 1024 tokens;
* the answer is the label sentence only ("\\n这篇帖子来自抑郁症社区。" + end token), because
  MATCHA posts cannot be sent to GPT-4 for explanations;
* after every epoch the dev subset is labelled by generation and the best epoch is kept (the
  paper selects on the validation set); then the test set is labelled with the best adapter.

The loss is computed from the decoder's hidden states at the answer positions only: GLM-4's
151,552-token vocabulary would otherwise need a full logits tensor (~2.5 GB at batch 4).
"""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from ..tasks import load_task
from ..utils import get_logger, save_json, set_seed
from .data import subset_indices
from .infer import build_prompt_ids, evaluate_texts, load_model
from .prompts import get_prompt

log = get_logger()
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_up_proj", "down_proj"]


def encode_example(tokenizer, prompt, text, label, hp) -> tuple[list[int], list[int]]:
    """(input_ids, labels) for one training example; labels are -100 on the prompt."""
    end = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    answer = tokenizer.encode("\n" + prompt.target(label), add_special_tokens=False) + [end]
    prompt_ids = build_prompt_ids(tokenizer, prompt, text, hp.get("max_text_tokens"), hp.get("max_text_chars"))
    room = hp["max_seq_len"] - len(answer)
    if len(prompt_ids) > room:  # keep the question's start and the chat-turn end
        prompt_ids = prompt_ids[:room - 1] + prompt_ids[-1:]
    return prompt_ids + answer, [-100] * len(prompt_ids) + answer


def collate(batch, pad_id: int):
    width = max(len(x) for x, _ in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, :len(x)], labels[i, :len(y)], mask[i, :len(x)] = torch.tensor(x), torch.tensor(y), 1
    return ids, mask, labels


def answer_loss(model, ids, mask, labels) -> torch.Tensor:
    """Mean cross-entropy over answer tokens, projecting only those positions to the vocabulary."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = base.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    target = labels[:, 1:]
    keep = target != -100
    logits = base.lm_head(hidden[:, :-1][keep]).float()
    return torch.nn.functional.cross_entropy(logits, target[keep])


def add_lora(model, hp):
    from peft import LoraConfig, get_peft_model

    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    config = LoraConfig(task_type="CAUSAL_LM", r=hp["lora_rank"], lora_alpha=hp["lora_alpha"],
                        lora_dropout=hp["lora_dropout"], target_modules=hp.get("lora_target") or LORA_TARGETS)
    model = get_peft_model(model, config)
    for p in model.parameters():  # adapters train in fp32 (fp16 autocast for the matmuls)
        if p.requires_grad:
            p.data = p.data.float()
    return model


def _split(splits, meta, name, k, seed):
    df = splits[name].reset_index(drop=True)
    stratify = None if meta["problem_type"].startswith("multi") else df["label"].astype(int).values
    idx = subset_indices(len(df), k, seed=seed, labels=stratify)
    return df.iloc[idx].reset_index(drop=True), idx


def _evaluate(model, tokenizer, prompt, df, meta, hp):
    model.eval()
    model.config.use_cache = True
    try:
        return evaluate_texts(model, tokenizer, prompt, df["text"].tolist(), df["label"].tolist(),
                              meta["label_names"], meta["average"], batch_size=hp["eval_batch_size"],
                              max_new_tokens=hp["max_new_tokens"], max_text_tokens=hp.get("max_text_tokens"),
                              max_text_chars=hp.get("max_text_chars"))
    finally:
        model.config.use_cache = False


def run_llm_finetune(task_root: Path, task: str, model_path: str, model_tag: str, output_root: Path,
                     seed: int, hp: dict, deadline: float | None = None, probe_steps: int | None = None) -> dict | None:
    """Train one (task, model, seed). With `probe_steps`, only time that many optimizer steps
    plus a small generation pass, and write probe.json instead of results."""
    out_dir = Path(output_root) / task / model_tag / f"seed{seed}"
    if probe_steps is None and (out_dir / "metrics.json").exists():
        log.info("Skipping %s / %s / seed %d (already done)", task, model_tag, seed)
        return None
    t_start = time.time()
    set_seed(seed)
    meta, splits = load_task(task_root, task)
    prompt = get_prompt(task)
    train_df, _ = _split(splits, meta, "train", hp.get("train_subset"), hp["subset_seed"])
    dev_df, _ = _split(splits, meta, "dev", hp.get("dev_subset"), hp["subset_seed"])
    test_df, test_idx = _split(splits, meta, "test", hp.get("test_subset"), hp["subset_seed"])

    model, tokenizer = load_model(model_path, hp["quant"])
    model = add_lora(model, hp)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    examples = [encode_example(tokenizer, prompt, t, y, hp) for t, y in zip(train_df["text"], train_df["label"])]
    log.info("%s: %d train / %d dev / %d test, %.1fM trainable, mean %d tokens", task, len(train_df), len(dev_df),
             len(test_df), trainable / 1e6, np.mean([len(x) for x, _ in examples]))

    bs, accum = hp["batch_size"], hp["grad_accum"]
    steps_per_epoch = math.ceil(len(examples) / (bs * accum))
    total_steps = probe_steps or steps_per_epoch * hp["epochs"]
    params = [p for p in model.parameters() if p.requires_grad]
    # float(): YAML reads "1e-4" (no decimal point) as a string
    optim = torch.optim.AdamW(params, lr=float(hp["learning_rate"]), weight_decay=float(hp.get("weight_decay") or 0.0))
    warmup = max(1, int(float(hp["warmup_ratio"]) * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: min((s + 1) / warmup, max(0.0, (total_steps - s) / max(1, total_steps - warmup))))
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    device = next(model.parameters()).device
    rng = np.random.default_rng(seed)
    pad = tokenizer.pad_token_id

    history, best, best_epoch, bad_epochs, step = [], -1.0, None, 0, 0
    best_dir = out_dir / "best_adapter"
    t_train = time.time()
    for epoch in range(1, (1 if probe_steps else hp["epochs"]) + 1):
        model.train()
        order = rng.permutation(len(examples))
        batches = [order[i:i + bs] for i in range(0, len(order), bs)]
        running, n_loss = 0.0, 0
        for b, chunk in enumerate(batches):
            if deadline and time.time() > deadline:
                raise SystemExit(f"{task}/{model_tag}/seed{seed}: deadline reached during training; rerun next session")
            ids, mask, labels = (t.to(device) for t in collate([examples[i] for i in chunk], pad))
            with torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                loss = answer_loss(model, ids, mask, labels)
            scaler.scale(loss / accum).backward()
            running, n_loss = running + loss.item(), n_loss + 1
            if (b + 1) % accum == 0 or b + 1 == len(batches):
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(params, hp.get("max_grad_norm", 1.0))
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % 10 == 0 or step == total_steps:
                    log.info("epoch %d step %d/%d loss %.4f lr %.2e (%.1f s/step)", epoch, step, total_steps,
                             running / n_loss, sched.get_last_lr()[0], (time.time() - t_train) / step)
                    running, n_loss = 0.0, 0
                if probe_steps and step >= probe_steps:
                    break
        if probe_steps:
            break
        dev = _evaluate(model, tokenizer, prompt, dev_df, meta, hp)
        f1 = dev["metrics"]["f1"]
        history.append({"epoch": epoch, "dev_f1": f1, "dev_unparsed": dev["metrics"]["unparsed"],
                        "minutes": round((time.time() - t_start) / 60, 1)})
        log.info("%s seed %d epoch %d: dev %s-F1 %.4f (unparsed %d)", task, seed, epoch, meta["average"], f1,
                 dev["metrics"]["unparsed"])
        if f1 > best:
            best, best_epoch, bad_epochs = f1, epoch, 0
            model.save_pretrained(best_dir)
        else:
            bad_epochs += 1
            if hp.get("early_stopping_patience") and bad_epochs >= hp["early_stopping_patience"]:
                break

    if probe_steps:
        sec_per_step = (time.time() - t_train) / max(step, 1)
        gen = _evaluate(model, tokenizer, prompt, dev_df.head(hp.get("probe_eval_items", 64)), meta, hp)
        probe = {"task": task, "model_tag": model_tag, "steps": step, "seconds_per_step": round(sec_per_step, 2),
                 "examples_per_step": bs * accum, "steps_per_epoch_full_subset": steps_per_epoch,
                 "minutes_per_epoch": round(sec_per_step * steps_per_epoch / 60, 1),
                 "mean_train_tokens": round(float(np.mean([len(x) for x, _ in examples])), 1),
                 "eval_items_per_second": gen["metrics"]["items_per_second"],
                 "minutes_dev_eval": round(len(dev_df) / gen["metrics"]["items_per_second"] / 60, 1),
                 "minutes_test_eval": round(len(test_df) / gen["metrics"]["items_per_second"] / 60, 1),
                 "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
                 "hyperparameters": hp}
        save_json(probe, Path(output_root) / "_probe" / f"{task}__{model_tag}.json")
        log.info("probe %s: %s", task, probe)
        return probe

    if best_epoch != history[-1]["epoch"]:  # restore the best epoch's adapter
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(str(best_dir / "adapter_model.safetensors")))
    test = _evaluate(model, tokenizer, prompt, test_df, meta, hp)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "test_predictions.npz", preds=test["preds"], labels=test["labels"],
                        indices=np.asarray(test_idx))
    with open(out_dir / "test_answers.jsonl", "w", encoding="utf-8") as f:
        for i, a in zip(test_idx, test["answers"]):
            f.write(json.dumps({"index": int(i), "answer": a}, ensure_ascii=False) + "\n")
    record = {
        "task": task, "model_tag": model_tag, "model_path": str(model_path), "seed": seed,
        "problem_type": meta["problem_type"], "average": meta["average"], "label_names": meta["label_names"],
        "hyperparameters": hp, "trainable_parameters": trainable,
        "n_train": len(train_df), "n_dev": len(dev_df), "n_test": len(test_df), "n_test_full": len(splits["test"]),
        "test_subset": len(test_df) < len(splits["test"]),
        "best_epoch": best_epoch, "history": history, "train_runtime": round(time.time() - t_train, 1),
        "minutes_total": round((time.time() - t_start) / 60, 1),
        "effective_batch_size": bs * accum, "n_gpu": 1,
        "test": test["metrics"],
    }
    save_json(record, out_dir / "metrics.json")
    if not hp.get("keep_adapter", True):
        shutil.rmtree(best_dir, ignore_errors=True)
    log.info("%s | %s | seed %d | test %s-F1 = %.4f (unparsed %d)", task, model_tag, seed, meta["average"],
             test["metrics"]["f1"], test["metrics"]["unparsed"])
    return record

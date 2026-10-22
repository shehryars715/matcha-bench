"""Fine-tune an encoder on one task with one seed, then evaluate on dev and test.

The protocol follows Chinese MentalBERT: Adam(W), lr 2e-5, batch 16, up to 10 epochs, and the
best dev checkpoint is evaluated on test. Additions: several seeds, optional early stopping
and class weighting (needed for imbalanced MATCHA tasks), and saved test predictions so
significance tests can be run afterwards. A model can bring its own source paper's recipe
(`model_overrides` in configs/finetune.yaml, e.g. the RoBERTa baseline's lr 1e-5 and dropout 0.3).
"""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from ..tasks import load_task
from ..utils import get_logger, save_json, set_seed
from .metrics import MULTI, classification_metrics, predictions_from_logits

log = get_logger()


class EncodedDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: dict, labels: list):
        self.encodings, self.labels = encodings, labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        item = {k: v[i] for k, v in self.encodings.items()}
        item["labels"] = self.labels[i]
        return item


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.class_weights is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device))
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def run_finetune(
    task_root: Path,
    task: str,
    model_path: str,
    model_tag: str,
    output_root: Path,
    seed: int,
    hp: dict,
) -> dict | None:
    out_dir = Path(output_root) / task / model_tag / f"seed{seed}"
    if (out_dir / "metrics.json").exists() and not hp.get("overwrite", False):
        log.info("Skipping %s / %s / seed %d (already done)", task, model_tag, seed)
        return None
    set_seed(seed)
    meta, splits = load_task(task_root, task)
    problem_type, label_names = meta["problem_type"], meta["label_names"]
    max_length = hp.get("max_length") or meta["max_length"]

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def encode(df):
        enc = tokenizer(df["text"].tolist(), truncation=True, max_length=max_length)
        if problem_type == MULTI:
            labels = [list(map(float, y)) for y in df["label"]]
        else:
            labels = [int(y) for y in df["label"]]
        return EncodedDataset(dict(enc), labels)

    datasets = {s: encode(df) for s, df in splits.items()}
    # dropout in front of the classification layer; unset keeps the model's own (hidden_dropout_prob, 0.1)
    dropout = {} if hp.get("classifier_dropout") is None else {"classifier_dropout": hp["classifier_dropout"]}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(label_names),
        problem_type=problem_type,
        id2label=dict(enumerate(label_names)),
        label2id={n: i for i, n in enumerate(label_names)},
        **dropout,
    )

    class_weights = None
    if hp.get("class_weight") == "balanced" and problem_type != MULTI:
        counts = np.bincount(np.asarray(datasets["train"].labels), minlength=len(label_names)).astype(float)
        class_weights = torch.tensor(counts.sum() / (len(label_names) * np.maximum(counts, 1)), dtype=torch.float)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = logits[0] if isinstance(logits, tuple) else logits
        m = classification_metrics(labels, predictions_from_logits(logits, problem_type), problem_type, meta["average"], label_names)
        return {k: v for k, v in m.items() if isinstance(v, float)}

    n_gpu = max(1, torch.cuda.device_count())
    steps_per_epoch = -(-len(datasets["train"]) // (hp["batch_size"] * n_gpu * hp.get("grad_accum", 1)))
    total_steps = steps_per_epoch * hp["epochs"]
    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=hp["epochs"],
        learning_rate=hp["learning_rate"],
        per_device_train_batch_size=hp["batch_size"],
        per_device_eval_batch_size=hp["batch_size"] * 2,
        gradient_accumulation_steps=hp.get("grad_accum", 1),
        warmup_steps=int(hp.get("warmup_ratio", 0.0) * total_steps),
        weight_decay=hp.get("weight_decay", 0.0),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=hp.get("dataloader_num_workers", 2),
        report_to="none",
        seed=seed,
    )
    # With DataParallel (2 GPUs) some transformers versions multiply the loss by n_gpu when
    # average_tokens_across_devices is on, although BERT's loss is already a per-replica mean.
    # Turning it off gives identical training on 1 or 2 GPUs.
    if any(f.name == "average_tokens_across_devices" for f in dataclasses.fields(TrainingArguments)):
        args.average_tokens_across_devices = False
    callbacks = [EarlyStoppingCallback(hp["early_stopping_patience"])] if hp.get("early_stopping_patience") else []
    trainer = WeightedLossTrainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["dev"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        class_weights=class_weights,
    )
    train_out = trainer.train()

    results = {}
    for split in ("dev", "test"):
        pred = trainer.predict(datasets[split])
        logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        preds = predictions_from_logits(logits, problem_type)
        labels = np.asarray(datasets[split].labels).astype(int)
        results[split] = classification_metrics(labels, preds, problem_type, meta["average"], label_names)
        if split == "test":
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_dir / "test_predictions.npz", logits=logits, preds=preds, labels=labels)

    record = {
        "task": task, "model_tag": model_tag, "model_path": str(model_path), "seed": seed,
        "problem_type": problem_type, "average": meta["average"], "label_names": label_names,
        "hyperparameters": {**hp, "max_length": max_length}, "train_runtime": train_out.metrics.get("train_runtime"),
        "n_gpu": n_gpu, "effective_batch_size": hp["batch_size"] * n_gpu * hp.get("grad_accum", 1),
        "epochs_trained": trainer.state.epoch,
        "dev": results["dev"], "test": results["test"],
    }
    save_json(record, out_dir / "metrics.json")
    if not hp.get("keep_checkpoints", False):
        shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)
    log.info("%s | %s | seed %d | test %s-F1 = %.4f", task, model_tag, seed, meta["average"], results["test"]["f1"])
    return record

"""Continued (domain-adaptive) MLM pre-training with lexicon-guided whole-word masking."""

from __future__ import annotations

import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    BertConfig,
    BertForMaskedLM,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from ..config import resolve_path
from ..lexicon import load_lexicon
from ..utils import get_logger, save_json, set_seed
from .collator import LexiconGuidedWWMCollator, select_mask_positions
from .corpus import ChunkDataset, lexicon_lookup, load_word_vocab

log = get_logger()


class DeadlineCallback(TrainerCallback):
    """Save a checkpoint and stop cleanly once a wall-clock deadline (epoch seconds) passes,
    so a time-limited session (e.g. Kaggle) ends with a resumable checkpoint, not a kill."""

    def __init__(self, deadline: float):
        self.deadline = deadline
        self.hit = False

    def on_step_end(self, args, state, control, **kwargs):
        if time.time() >= self.deadline:
            self.hit = True
            control.should_save = True
            control.should_training_stop = True
        return control


def lexicon_coverage(dataset: ChunkDataset, mlm_probability: float, n_samples: int = 20000, seed: int = 0) -> dict:
    """How much of the masking budget the lexicon actually fills (reported in the paper)."""
    rng = random.Random(seed)
    idx = rng.sample(range(len(dataset)), min(n_samples, len(dataset)))
    with_lex, lex_tokens, lex_masked, masked_total = 0, 0, 0, 0
    for i in idx:
        ex = dataset[i]
        lex = ex["lexicon"]
        sel = select_mask_positions(ex["word_start"], lex, mlm_probability, rng=rng)
        n_lex = sum(lex)
        with_lex += n_lex > 0
        lex_tokens += n_lex
        lex_masked += sum(1 for s, l in zip(sel, lex) if s and l)
        masked_total += sum(sel)
    n = max(len(idx), 1)
    return {
        "chunks_sampled": len(idx),
        "chunks_with_lexicon_word": with_lex / n,
        "lexicon_tokens_per_chunk": lex_tokens / n,
        "share_of_masked_tokens_from_lexicon": lex_masked / max(masked_total, 1),
    }


def _tiny_model(tokenizer) -> BertForMaskedLM:
    config = BertConfig(
        vocab_size=len(tokenizer), hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=128, max_position_embeddings=512,
    )
    return BertForMaskedLM(config)


def run_dapt(cfg: dict) -> dict:
    set_seed(cfg.get("seed", 42))
    out_dir = resolve_path(cfg["output_dir"])
    cache_dir = resolve_path(cfg["corpus_cache"])
    tokenizer = AutoTokenizer.from_pretrained(cfg.get("tokenizer_name") or cfg["model_name_or_path"])

    words = load_word_vocab(cache_dir / "words.json")
    lexicon_path = cfg.get("lexicon")
    lexicon = load_lexicon(resolve_path(lexicon_path)) if lexicon_path else set()
    lookup = lexicon_lookup(words, lexicon)
    info = {
        "lexicon": lexicon_path,
        "lexicon_size": len(lexicon),
        "lexicon_words_in_corpus": int(lookup.sum()) if lookup is not None else 0,
        "lexicon_order": cfg.get("lexicon_order", "sequential"),
        "mlm_probability": cfg.get("mlm_probability", 0.2),
    }

    train_ds = ChunkDataset(cache_dir / "train.npz", tokenizer.cls_token_id, tokenizer.sep_token_id, lookup, cfg.get("max_train_chunks"))
    eval_path = cache_dir / "val.npz"
    eval_ds = (
        ChunkDataset(eval_path, tokenizer.cls_token_id, tokenizer.sep_token_id, lookup, cfg.get("max_eval_chunks", 5000))
        if eval_path.exists() else None
    )
    info["train_chunks"] = len(train_ds)
    info["coverage"] = lexicon_coverage(train_ds, info["mlm_probability"]) if lookup is not None else None
    log.info("DAPT setup: %s", info)

    model = _tiny_model(tokenizer) if cfg.get("tiny_model") else AutoModelForMaskedLM.from_pretrained(cfg["model_name_or_path"])
    collator = LexiconGuidedWWMCollator(tokenizer, info["mlm_probability"], info["lexicon_order"])

    t = cfg["training"]
    n_gpu = max(1, torch.cuda.device_count())
    per_device = t["per_device_train_batch_size"]
    accum = max(1, math.ceil(t["effective_batch_size"] / (per_device * n_gpu)))
    steps_per_epoch = math.ceil(len(train_ds) / (per_device * n_gpu * accum))
    total_steps = t["max_steps"] if t.get("max_steps", -1) > 0 else steps_per_epoch * t["num_train_epochs"]
    info.update(n_gpu=n_gpu, gradient_accumulation_steps=accum, total_steps=total_steps)

    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=per_device,
        per_device_eval_batch_size=per_device,
        gradient_accumulation_steps=accum,
        learning_rate=t["learning_rate"],
        weight_decay=t.get("weight_decay", 0.0),
        num_train_epochs=t["num_train_epochs"],
        max_steps=t.get("max_steps", -1),
        warmup_steps=int(t.get("warmup_ratio", 0.0) * total_steps),
        logging_steps=t.get("logging_steps", 100),
        save_strategy="steps",
        save_steps=t.get("save_steps", 1000),
        save_total_limit=t.get("save_total_limit", 2),
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=t.get("eval_steps", 1000),
        prediction_loss_only=True,
        fp16=torch.cuda.is_available() and t.get("fp16", True),
        dataloader_num_workers=t.get("dataloader_num_workers", 2),
        remove_unused_columns=False,
        report_to="none",
        seed=cfg.get("seed", 42),
    )
    deadline = DeadlineCallback(float(cfg["deadline"])) if cfg.get("deadline") else None
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds, data_collator=collator,
        callbacks=[deadline] if deadline else None,
    )
    last = get_last_checkpoint(str(out_dir)) if out_dir.exists() else None
    if last:
        log.info("Resuming from %s", last)
    train_result = trainer.train(resume_from_checkpoint=last)
    info["train_metrics"] = train_result.metrics

    if deadline and deadline.hit:
        info.update(completed=False, global_step=trainer.state.global_step)
        save_json({"config": cfg, **info}, Path(out_dir) / "progress.json")
        log.info("Deadline reached at step %d/%d; checkpoint saved, resume next session",
                 trainer.state.global_step, total_steps)
        return info

    if eval_ds is not None:
        metrics = trainer.evaluate()
        # On n_gpu > 1 with average_tokens_across_devices, transformers reports the loss x n_gpu
        # (training itself is left unchanged so all DAPT variants stay comparable).
        scale = n_gpu if n_gpu > 1 and getattr(args, "average_tokens_across_devices", False) else 1
        metrics["loss_scale"] = scale
        if metrics.get("eval_loss") is not None:
            metrics["eval_loss_true"] = metrics["eval_loss"] / scale
            metrics["perplexity"] = float(np.exp(metrics["eval_loss_true"]))
        info["eval_metrics"] = metrics

    final = Path(out_dir) / "final"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    info["completed"] = True
    save_json({"config": cfg, **info}, final / "run_info.json")
    (Path(out_dir) / "progress.json").unlink(missing_ok=True)
    log.info("Saved DAPT model to %s", final)
    return info

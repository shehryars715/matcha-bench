"""Load a (converted) GLM model and classify texts by generating MentalGLM-style answers.

Generation is greedy (MentalGLM's demo uses top_k=1) and stops at the first "。", because only the
first sentence is parsed; with greedy decoding that sentence is the same as in a full-length
answer. Prompts are batched with left padding, longest first, so a too-large batch fails at once.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..finetune.metrics import MULTI, classification_metrics
from ..utils import get_logger
from .prompts import TaskPrompt, parse_answer

log = get_logger()


def load_model(path: str, quant: str = "fp16", lora_path: str | None = None):
    """quant: "fp16" (spread over all visible GPUs), "4bit" (NF4, one GPU) or "fp32" (CPU tests)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.padding_side = "left"
    kwargs = {}
    if quant == "4bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16)
        kwargs["device_map"] = {"": 0}
        kwargs["dtype"] = torch.float16
    elif quant == "fp16":
        kwargs["device_map"] = "auto"
        kwargs["dtype"] = torch.float16
    elif quant == "fp32":
        kwargs["dtype"] = torch.float32
    else:
        raise ValueError(f"unknown quant {quant!r}")
    model = _from_pretrained(AutoModelForCausalLM, path, kwargs)
    if lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return model, tokenizer


def _from_pretrained(cls, path, kwargs):
    try:
        return cls.from_pretrained(path, **kwargs)
    except TypeError:  # transformers 4.x names the argument torch_dtype
        kwargs = dict(kwargs)
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        return cls.from_pretrained(path, **kwargs)


def truncate_text(text: str, tokenizer, max_tokens: int | None, max_chars: int | None) -> str:
    if max_chars:
        text = text[:max_chars]
    if max_tokens:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > max_tokens:
            text = tokenizer.decode(ids[:max_tokens])
    return text


def build_prompt_ids(tokenizer, prompt: TaskPrompt, text: str, max_text_tokens=None, max_text_chars=None) -> list[int]:
    user = prompt.user_turn(truncate_text(text, tokenizer, max_text_tokens, max_text_chars))
    ids = tokenizer.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, tokenize=True)
    if hasattr(ids, "keys"):  # BatchEncoding in newer transformers
        ids = ids["input_ids"]
    return list(ids)


class _StopAtPeriod:
    """Stopping criterion: every sequence has produced "。" (or an end token) since the prompt."""

    def __init__(self, tokenizer, prompt_len: int):
        self.tokenizer, self.prompt_len = tokenizer, prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        texts = self.tokenizer.batch_decode(input_ids[:, self.prompt_len:], skip_special_tokens=False)
        done = [("。" in t) or ("<|user|>" in t) or ("<|endoftext|>" in t) for t in texts]
        return torch.tensor(done, device=input_ids.device)


@torch.no_grad()
def generate_answers(model, tokenizer, prompt_ids: list[list[int]], batch_size: int = 8,
                     max_new_tokens: int = 48) -> list[str]:
    """Greedy answers for already-tokenised prompts, in input order."""
    from transformers import StoppingCriteriaList

    order = sorted(range(len(prompt_ids)), key=lambda i: -len(prompt_ids[i]))
    answers: list[str] = [""] * len(prompt_ids)
    pad = tokenizer.pad_token_id
    device = next(model.parameters()).device
    cuda = device.type == "cuda"

    def run(idx):
        width = max(len(prompt_ids[i]) for i in idx)
        ids = torch.full((len(idx), width), pad, dtype=torch.long)
        mask = torch.zeros((len(idx), width), dtype=torch.long)
        for row, i in enumerate(idx):
            seq = prompt_ids[i]
            ids[row, width - len(seq):] = torch.tensor(seq)
            mask[row, width - len(seq):] = 1
        # fp16 autocast as in training: LoRA adapters are kept in fp32 and would otherwise double
        # the activation memory of every adapted layer
        with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
            out = model.generate(
                input_ids=ids.to(device), attention_mask=mask.to(device), max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=pad,
                stopping_criteria=StoppingCriteriaList([_StopAtPeriod(tokenizer, width)]))
        for row, i in enumerate(idx):
            answers[i] = tokenizer.decode(out[row, width:], skip_special_tokens=True)

    pending = [order[s:s + batch_size] for s in range(0, len(order), batch_size)]
    while pending:
        idx = pending.pop(0)
        try:
            run(idx)
        except torch.OutOfMemoryError:
            if len(idx) == 1:
                raise
            torch.cuda.empty_cache()  # long inputs: retry this batch in two halves
            log.warning("out of GPU memory on a batch of %d; retrying in halves", len(idx))
            half = len(idx) // 2
            pending[:0] = [idx[:half], idx[half:]]
    return answers


def evaluate_texts(model, tokenizer, prompt: TaskPrompt, texts: list[str], labels, label_names: list[str],
                   average: str, batch_size: int = 8, max_new_tokens: int = 48,
                   max_text_tokens: int | None = None, max_text_chars: int | None = None) -> dict:
    """Generate, parse and score. Returns metrics plus per-item answers and predictions."""
    t0 = time.time()
    prompt_ids = [build_prompt_ids(tokenizer, prompt, t, max_text_tokens, max_text_chars) for t in texts]
    answers = generate_answers(model, tokenizer, prompt_ids, batch_size, max_new_tokens)
    seconds = time.time() - t0
    parsed = [parse_answer(a, prompt) for a in answers]
    problem_type = MULTI if prompt.multi_label else "single_label_classification"
    labels = np.asarray([list(map(int, y)) for y in labels] if prompt.multi_label else [int(y) for y in labels])
    preds = np.asarray(parsed)
    if prompt.multi_label:
        unparsed = int(sum(sum(p) == 0 for p in parsed))  # no distortion named
        metrics = classification_metrics(labels, preds, problem_type, average, label_names)
    else:
        unparsed = int((preds == -1).sum())
        metrics = classification_metrics(labels, preds, problem_type, average, label_names,
                                         labels_list=list(range(len(label_names))))
    metrics.update({
        "n": len(texts), "unparsed": unparsed, "seconds": round(seconds, 1),
        "items_per_second": round(len(texts) / max(seconds, 1e-9), 3),
        "mean_prompt_tokens": round(float(np.mean([len(p) for p in prompt_ids])), 1),
        "max_prompt_tokens": int(max(len(p) for p in prompt_ids)),
    })
    return {"metrics": metrics, "answers": answers, "preds": preds, "labels": labels}

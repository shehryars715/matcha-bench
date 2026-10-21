"""LLM baseline: prompts and parsing, checkpoint conversion, subsets, metrics with unparsed answers,
and the answer-only training loss. No network, tokenizer download or GPU needed."""

import numpy as np
import pytest
import torch

from matcha_bench.finetune.metrics import classification_metrics
from matcha_bench.llm.convert import convert_tensors, glm_config_from_chatglm
from matcha_bench.llm.data import subset_indices
from matcha_bench.llm.prompts import PROMPTS, get_prompt, parse_answer


@pytest.mark.parametrize("task", sorted(PROMPTS))
def test_every_label_round_trips_through_its_answer(task):
    p = get_prompt(task)
    for name in p.verbalizers:  # no name contains another, so parsing is unambiguous
        assert sum(name in other for other in p.verbalizers) == 1
    if p.multi_label:
        label = [1, 0] * (len(p.verbalizers) // 2)
        assert parse_answer("\n" + p.target(label) + "理由：……", p) == label
    else:
        for i in range(len(p.verbalizers)):
            assert parse_answer("\n" + p.target(i) + "理由：……", p) == i


def test_parsing_reads_the_first_sentence_only():
    p = get_prompt("suicide")
    assert parse_answer("\n这篇帖子体现了高自杀风险。理由：并非低自杀风险。", p) == 1
    assert parse_answer("无法判断。这篇帖子体现了高自杀风险。", p) == -1  # label outside sentence 1
    assert parse_answer("", p) == -1
    cd = get_prompt("cognitive")
    assert parse_answer("这篇帖子体现了心理过滤, 放大, 乱贴标签的认知歪曲。", cd) == \
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0]


def test_suicide_prompt_matches_the_released_evaluation_format():
    # Evaluation/LLM/generated_predictions.jsonl in the MentalGLM repository
    assert get_prompt("suicide").user_turn("帖子") == "考虑以下这个帖子，帖子体现了高自杀风险还是低自杀风险？\n帖子"


def _chatglm_config(**kw):
    cfg = {"model_type": "chatglm", "num_attention_heads": 4, "kv_channels": 8, "hidden_size": 32,
           "multi_query_attention": True, "multi_query_group_num": 2, "padded_vocab_size": 50,
           "ffn_hidden_size": 64, "num_layers": 2, "layernorm_epsilon": 1e-5, "rope_ratio": 500,
           "seq_length": 1024, "original_rope": True, "add_qkv_bias": True}
    cfg.update(kw)
    return cfg


def test_config_takes_rope_base_from_rope_ratio():
    assert glm_config_from_chatglm(_chatglm_config())["rope_theta"] == 5_000_000.0  # chat models
    assert glm_config_from_chatglm(_chatglm_config(rope_ratio=1))["rope_theta"] == 10_000.0  # base models
    cfg = glm_config_from_chatglm(_chatglm_config())
    assert (cfg["num_key_value_heads"], cfg["head_dim"], cfg["partial_rotary_factor"]) == (2, 8, 0.5)


def test_query_key_value_is_split_into_q_k_v():
    cfg = glm_config_from_chatglm(_chatglm_config())
    q, k, v = torch.randn(32, 32), torch.randn(16, 32), torch.randn(16, 32)  # 4 x 8 query, 2 x 8 key/value
    out = convert_tensors({
        "transformer.encoder.layers.1.self_attention.query_key_value.weight": torch.cat([q, k, v]),
        "transformer.encoder.layers.1.self_attention.query_key_value.bias": torch.arange(64.0),
        "transformer.encoder.layers.1.mlp.dense_h_to_4h.weight": torch.zeros(128, 32),
        "transformer.output_layer.weight": torch.zeros(50, 32),
        "transformer.rotary_pos_emb.inv_freq": torch.zeros(4),
    }, cfg)
    assert torch.equal(out["model.layers.1.self_attn.q_proj.weight"], q)
    assert torch.equal(out["model.layers.1.self_attn.k_proj.weight"], k)
    assert torch.equal(out["model.layers.1.self_attn.v_proj.weight"], v)
    assert torch.equal(out["model.layers.1.self_attn.v_proj.bias"], torch.arange(48.0, 64.0))
    assert out["model.layers.1.mlp.gate_up_proj.weight"].shape == (128, 32)
    assert "lm_head.weight" in out and not any("inv_freq" in k for k in out)
    with pytest.raises(KeyError):
        convert_tensors({"transformer.something_new": torch.zeros(1)}, cfg)


def test_subsets_are_fixed_exact_and_stratified():
    y = np.array([0] * 700 + [1] * 200 + [2] * 90 + [3] * 10)
    idx = subset_indices(len(y), 100, seed=0, labels=y)
    assert len(idx) == 100 and np.all(np.diff(idx) > 0)
    assert np.bincount(y[idx]).tolist() == [70, 20, 9, 1]
    assert np.array_equal(idx, subset_indices(len(y), 100, seed=0, labels=y))
    assert len(subset_indices(len(y), 8, seed=0, labels=y)) == 8  # every class still present
    assert np.array_equal(subset_indices(10, None), np.arange(10))


def test_unparsed_answers_count_as_wrong_without_becoming_a_class():
    y = np.array([0, 1, 2, 2])
    m = classification_metrics(y, np.array([0, 1, 2, -1]), "single_label_classification", "macro",
                               ["a", "b", "c"], labels_list=[0, 1, 2])
    assert m["accuracy"] == 0.75
    assert m["f1"] == pytest.approx((1 + 1 + 2 / 3) / 3)  # class c: recall 1/2, precision 1
    b = classification_metrics(np.array([1, 1, 0]), np.array([1, -1, 0]), "single_label_classification",
                               "binary", ["neg", "pos"])
    assert b["recall"] == 0.5 and b["precision"] == 1.0


def test_answer_only_loss_equals_the_full_causal_lm_loss():
    from transformers import GlmConfig, GlmForCausalLM

    from matcha_bench.llm.train import answer_loss, collate

    torch.manual_seed(0)
    model = GlmForCausalLM(GlmConfig(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                     num_attention_heads=4, num_key_value_heads=2, head_dim=8, pad_token_id=0))
    batch = [([5, 6, 7, 8, 9, 10], [-100, -100, -100, 8, 9, 10]), ([11, 12, 13, 14], [-100, -100, 13, 14])]
    ids, mask, labels = collate(batch, pad_id=0)
    assert labels[1, 4:].eq(-100).all() and mask[1].tolist() == [1, 1, 1, 1, 0, 0]
    with torch.no_grad():
        full = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        ours = answer_loss(model, ids, mask, labels)
    assert ours.item() == pytest.approx(full.item(), rel=1e-5)


def test_generation_halves_a_batch_that_runs_out_of_memory():
    from matcha_bench.llm.infer import generate_answers

    class Tok:
        pad_token_id = 0

        def batch_decode(self, rows, skip_special_tokens=False):
            return ["。"] * len(rows)

        def decode(self, ids, skip_special_tokens=True):
            return f"answer {int(ids[0])}"

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))
            self.sizes = []

        def generate(self, input_ids, attention_mask, **kw):
            self.sizes.append(len(input_ids))
            if len(input_ids) > 2:
                raise torch.OutOfMemoryError("fake")
            first = input_ids[:, -1:]  # echo each prompt's last token as the "answer"
            return torch.cat([input_ids, first], dim=1)

    model = Model()
    prompts = [[1, 2, i] for i in range(10, 17)]  # 7 prompts, batch 4 -> halves of 2
    answers = generate_answers(model, Tok(), prompts, batch_size=4)
    assert answers == [f"answer {i}" for i in range(10, 17)]  # every item answered, in order
    assert max(s for s in model.sizes if s <= 2) == 2 and 4 in model.sizes


def test_a_smaller_test_is_archived_and_training_state_kept(tmp_path):
    import json

    from matcha_bench.llm.train import archive_smaller_test

    run = tmp_path / "seed42"
    (run / "best_adapter").mkdir(parents=True)
    (run / "best_adapter" / "adapter_model.safetensors").write_bytes(b"x")
    history = [{"epoch": 1, "dev_f1": 0.58}, {"epoch": 2, "dev_f1": 0.52}]
    (run / "metrics.json").write_text(json.dumps({"n_test": 2000, "best_epoch": 1, "history": history}))
    (run / "test_predictions.npz").write_bytes(b"p")

    assert archive_smaller_test(run, 20000) is True
    assert not (run / "metrics.json").exists()
    assert (run / "test_subset2000" / "metrics.json").exists()
    assert (run / "test_subset2000" / "test_predictions.npz").exists()
    state = json.loads((run / "train_state.json").read_text())
    assert state["trained"] and state["best_epoch"] == 1 and state["history"] == history

    (run / "metrics.json").write_text(json.dumps({"n_test": 20000, "best_epoch": 1, "history": history}))
    assert archive_smaller_test(run, 20000) is False  # already full: left alone


# save points come after epoch 1's training (1) and dev check (2), epoch 2's (3, 4), epoch 3's training (5)
@pytest.mark.parametrize("stop_after_save", [1, 2, 3, 5])
def test_training_resumes_from_the_epoch_save_point(tmp_path, monkeypatch, stop_after_save):
    """3 epochs straight == stopped at a save point, then resumed: same weights and dev history."""
    import json

    pytest.importorskip("peft")
    from transformers import AutoTokenizer, GlmConfig, GlmForCausalLM

    from matcha_bench.llm import train

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    except Exception:
        pytest.skip("needs a cached chat tokenizer")
    model_dir = tmp_path / "model"
    torch.manual_seed(0)
    GlmForCausalLM(GlmConfig(vocab_size=len(tok), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                             num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                             pad_token_id=tok.pad_token_id)).save_pretrained(model_dir)
    tok.save_pretrained(model_dir)

    task = "matcha_user_binary_masked"
    task_dir = tmp_path / "tasks" / task
    task_dir.mkdir(parents=True)
    (task_dir / "meta.json").write_text(json.dumps({"problem_type": "single_label_classification",
                                                    "label_names": ["control", "disorder"], "average": "macro"}))
    for split, n in [("train", 12), ("dev", 4), ("test", 4)]:
        rows = [{"id": f"{split}{i}", "text": f"今天{i}天气", "label": i % 2} for i in range(n)]
        (task_dir / f"{split}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    hp = {"quant": "fp32", "lora_rank": 4, "lora_alpha": 8, "lora_dropout": 0.0, "lora_target": None,
          "learning_rate": 1e-2, "warmup_ratio": 0.01, "weight_decay": 0.0, "max_grad_norm": 1.0,
          "batch_size": 2, "grad_accum": 2, "epochs": 3, "early_stopping_patience": None, "max_seq_len": 128,
          "max_text_chars": 20, "max_new_tokens": 2, "eval_batch_size": 4, "subset_seed": 0,
          "train_subset": None, "dev_subset": None, "test_subset": None, "keep_adapter": True}

    finals = {}

    def fake_test_and_save(model, *args, **kw):  # record the trained adapter instead of testing
        out_dir = args[-7]
        finals[out_dir.parents[2].name] = {k: v.detach().clone() for k, v in model.named_parameters()
                                              if v.requires_grad}
        assert not (out_dir / "checkpoint").exists()  # removed once training is over
        return {}

    monkeypatch.setattr(train, "_test_and_save", fake_test_and_save)
    run = lambda root: train.run_llm_finetune(tmp_path / "tasks", task, str(model_dir), "m", tmp_path / root, 42, hp)
    run("straight")

    real_save = train._save_checkpoint
    saves = []

    def save_then_stop(ckpt_dir, *a, **kw):  # the session ends right after the chosen save point
        real_save(ckpt_dir, *a, **kw)
        saves.append(1)
        if len(saves) == stop_after_save:
            raise SystemExit("session over")

    monkeypatch.setattr(train, "_save_checkpoint", save_then_stop)
    with pytest.raises(SystemExit):
        run("resumed")
    assert (tmp_path / "resumed" / task / "m" / "seed42" / "checkpoint" / "state.pt").exists()
    monkeypatch.setattr(train, "_save_checkpoint", real_save)
    run("resumed")

    a, b = finals["straight"], finals["resumed"]
    assert a.keys() == b.keys() and len(a) > 0
    assert all(torch.equal(a[k], b[k]) for k in a)
    state = lambda root: json.loads((tmp_path / root / task / "m" / "seed42" / "train_state.json").read_text())
    assert [h["epoch"] for h in state("resumed")["history"]] == [1, 2, 3]
    assert [h["dev_f1"] for h in state("resumed")["history"]] == [h["dev_f1"] for h in state("straight")["history"]]


def test_generation_stops_between_batches_at_the_deadline():
    import time

    from matcha_bench.llm.infer import generate_answers

    with pytest.raises(SystemExit):
        generate_answers(torch.nn.Linear(1, 1), type("T", (), {"pad_token_id": 0})(), [[1, 2]],
                         deadline=time.time() - 1)


def test_zeroshot_scores_the_full_test_set_with_the_untrained_model(tmp_path):
    """A zero-shot run writes the same files as a fine-tuned one, with no training and no adapter."""
    import json

    import numpy as np
    from transformers import AutoTokenizer, GlmConfig, GlmForCausalLM

    from matcha_bench.llm import train

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    except Exception:
        pytest.skip("needs a cached chat tokenizer")
    model_dir = tmp_path / "model"
    torch.manual_seed(0)
    GlmForCausalLM(GlmConfig(vocab_size=len(tok), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                             num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                             pad_token_id=tok.pad_token_id)).save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    task = "matcha_user_binary_masked"
    task_dir = tmp_path / "tasks" / task
    task_dir.mkdir(parents=True)
    (task_dir / "meta.json").write_text(json.dumps({"problem_type": "single_label_classification",
                                                    "label_names": ["control", "disorder"], "average": "macro"}))
    for split, n in [("train", 6), ("dev", 2), ("test", 5)]:
        rows = [{"id": f"{split}{i}", "text": f"今天{i}天气", "label": i % 2} for i in range(n)]
        (task_dir / f"{split}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    hp = {"quant": "fp32", "max_text_chars": 20, "max_new_tokens": 2, "eval_batch_size": 4, "subset_seed": 0,
          "test_subset": None, "keep_adapter": True, "batch_size": 4, "grad_accum": 8}

    run = lambda: train.run_llm_zeroshot(tmp_path / "tasks", task, str(model_dir), "m", tmp_path / "out", 0, hp)
    record = run()
    out = tmp_path / "out" / task / "m" / "seed0"
    assert record["n_test"] == 5 and record["n_train"] == 0 and record["trainable_parameters"] == 0
    assert record["best_epoch"] is None and record["test"]["n"] == 5
    assert np.load(out / "test_predictions.npz")["labels"].tolist() == [0, 1, 0, 1, 0]
    assert not (out / "best_adapter").exists()
    assert run() is None  # finished runs are skipped

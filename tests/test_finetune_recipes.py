"""Per-model fine-tuning recipes (model_overrides) and the RoBERTa baseline's sanity-check task."""

import json
from pathlib import Path

import pandas as pd
import pytest

from matcha_bench.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def hparams(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from run_finetune import hparams

    return hparams


def test_model_recipe_overrides_the_task_settings_but_keeps_its_class_weights(hparams):
    cfg = load_config(ROOT / "configs" / "finetune.yaml")
    bert = hparams(cfg, "matcha_user_multiclass_masked", "bert-wwm-ext")
    roberta = hparams(cfg, "matcha_user_multiclass_masked", "roberta-wwm-ext")
    assert (bert["epochs"], bert["learning_rate"], bert["early_stopping_patience"]) == (6, 2e-5, 2)
    assert bert["classifier_dropout"] is None
    assert (roberta["epochs"], roberta["learning_rate"], roberta["early_stopping_patience"]) == (10, 1e-5, None)
    assert (roberta["classifier_dropout"], roberta["max_length"], roberta["batch_size"]) == (0.3, 512, 16)
    assert roberta["class_weight"] == bert["class_weight"] == "balanced"
    assert hparams(cfg, "reddit_posts", "roberta-base")["class_weight"] is None  # balanced data, plain loss


def test_roberta_pipeline_uses_known_models_and_the_reddit_task_root():
    pipe = load_config(ROOT / "configs" / "pipeline_roberta.yaml")
    for stage in (s for s in pipe["stages"] if s["type"] == "finetune"):
        ft = load_config(ROOT / stage["config"])
        assert set(stage["models"]) <= set(ft["models"]) and set(stage["models"]) <= set(ft["model_overrides"])
    reddit = load_config(ROOT / "configs" / "finetune_reddit.yaml")
    assert reddit["task_root"] == "outputs/tasks"
    assert reddit["output_root"] == load_config(ROOT / "configs" / "finetune.yaml")["output_root"]


@pytest.mark.parametrize("dropout, expected", [(0.3, 0.3), (None, 0.1)])
def test_classifier_dropout_reaches_the_model(tmp_path, monkeypatch, tiny_tokenizer, dropout, expected):
    from transformers import AutoModelForSequenceClassification, BertConfig, BertForMaskedLM

    from matcha_bench.finetune import run
    from matcha_bench.tasks import TaskData, save_task

    model_dir = tmp_path / "tiny-bert"
    config = BertConfig(vocab_size=len(tiny_tokenizer), hidden_size=16, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=32, max_position_embeddings=64)
    BertForMaskedLM(config).save_pretrained(model_dir)
    tiny_tokenizer.save_pretrained(model_dir)
    texts = ["我最近很绝望", "今天天气好", "失眠焦虑", "医生说没事的"] * 3
    frame = pd.DataFrame({"text": texts, "label": [1, 0, 1, 0] * 3})
    save_task(TaskData(name="toy", splits={"train": frame, "dev": frame, "test": frame},
                       label_names=["calm", "distress"], max_length=32), tmp_path / "tasks")

    loaded = []

    class Recording:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            loaded.append(AutoModelForSequenceClassification.from_pretrained(*args, **kwargs))
            return loaded[-1]

    monkeypatch.setattr(run, "AutoModelForSequenceClassification", Recording)
    hp = {"epochs": 1, "learning_rate": 1e-3, "batch_size": 4, "early_stopping_patience": None,
          "class_weight": None, "max_length": None, "classifier_dropout": dropout, "dataloader_num_workers": 0}
    run.run_finetune(tmp_path / "tasks", "toy", str(model_dir), "tiny", tmp_path / "out", 42, hp)

    assert loaded[0].dropout.p == pytest.approx(expected)
    metrics = json.loads((tmp_path / "out" / "toy" / "tiny" / "seed42" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["hyperparameters"]["classifier_dropout"] == dropout
    assert metrics["hyperparameters"]["max_length"] == 32  # unset -> the task's own


def write_reddit_csvs(folder: Path, class_ids=None) -> None:
    folder.mkdir(parents=True)
    names = ["adhd", "anxiety", "bipolar", "depression", "ptsd", "none"]
    for split in ("train", "val", "test"):
        pd.DataFrame({
            "ID": [f"{split}{i}" for i in range(6)],
            "post": [f"post {i}" for i in range(5)] + [None],
            "class_name": names,
            "class_id": class_ids or list(range(6)),
        }).to_csv(folder / f"posts_{split}.csv", index=False)


def test_reddit_task_reads_the_authors_split_files(tmp_path):
    from matcha_bench.tasks.reddit_tasks import LABELS, build_reddit_posts

    write_reddit_csvs(tmp_path / "drive" / "data")  # found in subfolders too
    task = build_reddit_posts(tmp_path / "drive")
    assert list(task.splits) == ["train", "dev", "test"]
    dev = task.splits["dev"]
    assert dev["id"].tolist()[0] == "val0" and dev["label"].tolist() == list(range(6))
    assert dev["text"].tolist()[-1] == ""  # an empty post stays (the authors' split is kept as is)
    assert (task.label_names, task.average, task.max_length) == (LABELS, "macro", 512)


def test_reddit_task_rejects_class_ids_out_of_the_documented_order(tmp_path):
    from matcha_bench.tasks.reddit_tasks import build_reddit_posts

    write_reddit_csvs(tmp_path / "drive", class_ids=[5, 0, 1, 2, 3, 4])
    with pytest.raises(ValueError, match="class_id"):
        build_reddit_posts(tmp_path / "drive")

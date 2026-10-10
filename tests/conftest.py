import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
CHARS = list("我最近很绝望想到死亡今天天气好失眠焦虑抑郁症吃药医生说没事的了，。")


@pytest.fixture(scope="session")
def tiny_tokenizer_dir(tmp_path_factory):
    """A character-level BERT tokenizer saved to disk (no network needed)."""
    from transformers import BertTokenizerFast

    d = tmp_path_factory.mktemp("tok")
    vocab = d / "vocab.txt"
    vocab.write_text("\n".join(SPECIALS + sorted(set(CHARS)) + ["emo", "##s"]) + "\n", encoding="utf-8")
    tok = BertTokenizerFast.from_pretrained(str(d))  # works on transformers 4.x and 5.x
    tok.save_pretrained(str(d))
    return d


@pytest.fixture(scope="session")
def tiny_tokenizer(tiny_tokenizer_dir):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(tiny_tokenizer_dir))

from matcha_bench.terms import DISORDER_PLACEHOLDER, DRUG_PLACEHOLDER, mask_shortcut_terms
from matcha_bench.text import clean_text, is_repost, is_truncated


def test_clean_removes_weibo_noise():
    raw = "#抑郁症超话# 今天又失眠了[泪] @小明 http://t.cn/abc 😭 ...全文"
    assert clean_text(raw) == "今天又失眠了"


def test_clean_removes_repost_chain_and_boilerplate():
    assert clean_text("转发微博//@某人:好难受") == ""
    assert clean_text("真好看 某某的微博视频") == "真好看"


def test_clean_handles_missing():
    assert clean_text(None) == ""
    assert clean_text("   ") == ""


def test_flags():
    assert is_repost("转发微博")
    assert is_repost("说得对//@abc: 原文")
    assert not is_repost("我今天很好")
    assert is_truncated("很长的一段话 ...全文")


def test_mask_terms_longest_match_and_drugs():
    text = "确诊精神分裂症后开始吃奥氮平，双相的朋友也在吃"
    masked = mask_shortcut_terms(text)
    assert "精神分裂" not in masked and "奥氮平" not in masked and "双相" not in masked
    assert masked.count(DISORDER_PLACEHOLDER) == 2 and DRUG_PLACEHOLDER in masked
    assert "确诊" in masked  # self-report cue is not a disorder name


def test_mask_terms_keeps_ambiguous_emotion_words_by_default():
    assert "焦虑" in mask_shortcut_terms("最近很焦虑")
    assert "焦虑" not in mask_shortcut_terms("最近很焦虑", include_ambiguous=True)


def test_mask_latin_abbreviations_whole_word_only():
    assert mask_shortcut_terms("my ADHD meds") == f"my {DISORDER_PLACEHOLDER} meds"
    assert mask_shortcut_terms("ADDRESS") == "ADDRESS"

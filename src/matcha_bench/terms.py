"""Shortcut-term masking.

MATCHA cohorts are defined by Super-Topic membership, so posts often name the disorder,
the community or a medication outright. To check whether a model reads how people describe
their experience, and not just these labels, the ``*_masked`` task variants replace
disorder names, community nicknames and psychiatric drug names with neutral placeholders.

Bare emotion words that double as symptom words (焦虑, 抑郁, 强迫, 多动, 自闭) are left
in deliberately: they describe experience and are not only diagnosis labels. Pass
``include_ambiguous=True`` to mask them as well.
"""

from __future__ import annotations

import re
from functools import lru_cache

DISORDER_TERMS: dict[str, list[str]] = {
    "ADHD": ["注意缺陷多动障碍", "注意力缺陷多动障碍", "注意力缺陷障碍", "多动症", "ADHD", "ADD"],
    "ASD": ["孤独症谱系障碍", "自闭症谱系障碍", "孤独症", "自闭症", "阿斯伯格综合征", "阿斯伯格", "星星的孩子", "星宝", "ASD", "autism"],
    # 双向 is a frequent homophone misspelling of 双相 in the BD community.
    "BD": ["双相情感障碍", "双相障碍", "双相", "双向情感障碍", "双向障碍", "双向", "躁郁症", "躁郁", "躁狂症", "bipolar"],
    "DEP": ["抑郁症", "抑郁障碍", "重度抑郁", "中度抑郁", "轻度抑郁", "郁友", "depression"],
    "GAD": ["广泛性焦虑障碍", "广泛性焦虑", "焦虑障碍", "焦虑症", "焦友", "惊恐障碍", "anxiety"],
    "OCD": ["强迫性障碍", "强迫障碍", "强迫症", "强友", "OCD"],
    "SZ": ["精神分裂症", "精神分裂", "思觉失调症", "思觉失调", "分裂症", "精分", "schizophrenia"],
}

AMBIGUOUS_TERMS = ["抑郁", "焦虑", "强迫", "多动", "自闭"]

DRUG_TERMS = [
    # SSRIs / SNRIs / other antidepressants (generic + common brand names in China)
    "舍曲林", "左洛复", "氟西汀", "百忧解", "帕罗西汀", "赛乐特", "艾司西酞普兰", "来士普", "西酞普兰",
    "氟伏沙明", "兰释", "文拉法辛", "怡诺思", "度洛西汀", "欣百达", "米氮平", "瑞美隆", "安非他酮",
    "阿戈美拉汀", "维度新", "伏硫西汀", "心达悦", "曲唑酮", "黛力新", "氟哌噻吨美利曲辛",
    # anxiolytics / hypnotics
    "坦度螺酮", "丁螺环酮", "劳拉西泮", "阿普唑仑", "氯硝西泮", "艾司唑仑", "奥沙西泮", "地西泮",
    "安定", "佐匹克隆", "右佐匹克隆", "唑吡坦", "思诺思",
    # mood stabilisers
    "碳酸锂", "锂盐", "丙戊酸钠", "丙戊酸镁", "德巴金", "拉莫三嗪", "利必通",
    # antipsychotics
    "喹硫平", "思瑞康", "奥氮平", "再普乐", "阿立哌唑", "安律凡", "博思清", "利培酮", "维思通",
    "帕利哌酮", "芮达", "氯氮平", "齐拉西酮", "鲁拉西酮", "氨磺必利", "舒必利",
    # ADHD medication
    "哌甲酯", "专注达", "利他林", "托莫西汀", "择思达",
]

DISORDER_PLACEHOLDER = "某病"
DRUG_PLACEHOLDER = "某药"


def _alternation(terms: list[str]) -> str:
    # Longest first so that 精神分裂症 wins over 精神分裂.
    return "|".join(re.escape(t) for t in sorted(set(terms), key=len, reverse=True))


@lru_cache(maxsize=4)
def _patterns(include_ambiguous: bool) -> tuple[re.Pattern, re.Pattern]:
    disorder_terms = [t for terms in DISORDER_TERMS.values() for t in terms]
    if include_ambiguous:
        disorder_terms += AMBIGUOUS_TERMS
    # Latin abbreviations must not fire inside longer words (e.g. "ADD" in "ADDRESS").
    latin = [t for t in disorder_terms if t.isascii()]
    cjk = [t for t in disorder_terms if not t.isascii()]
    disorder_re = re.compile(
        rf"(?:{_alternation(cjk)})|(?<![A-Za-z])(?:{_alternation(latin)})(?![A-Za-z])", re.IGNORECASE
    )
    drug_re = re.compile(_alternation(DRUG_TERMS))
    return disorder_re, drug_re


def mask_shortcut_terms(text: str, *, mask_drugs: bool = True, include_ambiguous: bool = False) -> str:
    disorder_re, drug_re = _patterns(include_ambiguous)
    text = disorder_re.sub(DISORDER_PLACEHOLDER, text)
    if mask_drugs:
        text = drug_re.sub(DRUG_PLACEHOLDER, text)
    return text


def all_shortcut_terms() -> list[str]:
    return sorted({t for terms in DISORDER_TERMS.values() for t in terms} | set(DRUG_TERMS))

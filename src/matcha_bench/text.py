"""Weibo text cleaning.

Follows the Chinese MentalBERT pre-processing (remove URLs, @mentions, #topic# tags,
emoticons and special symbols) and adds Weibo-specific clean-up observed in MATCHA:
truncation markers ("...全文"), repost chains ("//@user: ..."), Super-Topic tags and
platform boilerplate ("转发微博", "XX的微博视频").
"""

from __future__ import annotations

import re

URL_RE = re.compile(r"(?:https?://|www\.)[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+")
REPOST_CHAIN_RE = re.compile(r"//\s*@.*$", re.S)
MENTION_RE = re.compile(r"@[\w一-鿿\-·]{1,30}:?")
TOPIC_RE = re.compile(r"#[^#\n]{1,80}#")
SUPER_TOPIC_RE = re.compile(r"[\w一-鿿]{1,20}超话")
WEIBO_EMOTICON_RE = re.compile(r"\[[一-鿿A-Za-z0-9_]{1,10}\]")
TRUNCATION_RE = re.compile(r"\s*(?:\.{2,}|…+)?\s*全文\s*$")
VIDEO_BOILERPLATE_RE = re.compile(r"\S{0,40}的微博视频")
EMOJI_RE = re.compile(
    "["
    "\U0001f000-\U0001faff"  # emoji, symbols, pictographs
    "\U00002600-\U000027bf"  # misc symbols, dingbats
    "\U0001f1e6-\U0001f1ff"  # flags
    "\U0000fe0f\U0000200d\U000020e3"  # variation selector, ZWJ, keycap
    "\U0000e000-\U0000f8ff"  # private use (platform icons)
    "]"
)
BOILERPLATE = ("转发微博", "分享图片", "网页链接", "查看图片", "评论配图", "Repost")
WHITESPACE_RE = re.compile(r"\s+")


def is_repost(raw: str) -> bool:
    if not isinstance(raw, str):
        return False
    stripped = raw.strip()
    return stripped.startswith("转发微博") or stripped.startswith("Repost") or "//@" in stripped


def is_truncated(raw: str) -> bool:
    return isinstance(raw, str) and bool(TRUNCATION_RE.search(raw))


def clean_text(raw: str, *, strip_topics: bool = True) -> str:
    """Return cleaned post text ("" for missing or empty content)."""
    if not isinstance(raw, str):
        return ""
    text = TRUNCATION_RE.sub("", raw)
    text = REPOST_CHAIN_RE.sub("", text)
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    if strip_topics:
        text = TOPIC_RE.sub("", text)
        text = SUPER_TOPIC_RE.sub("", text)
    text = WEIBO_EMOTICON_RE.sub("", text)
    text = EMOJI_RE.sub("", text)
    text = VIDEO_BOILERPLATE_RE.sub("", text)
    for phrase in BOILERPLATE:
        text = text.replace(phrase, "")
    return WHITESPACE_RE.sub(" ", text).strip()


def is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2A6DF
        or 0x2A700 <= cp <= 0x2CEAF
        or 0xF900 <= cp <= 0xFAFF
        or 0x2F800 <= cp <= 0x2FA1F
    )


def is_all_cjk(word: str) -> bool:
    return bool(word) and all(is_cjk(c) for c in word)

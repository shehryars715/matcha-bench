"""Mining candidate lexicon words from MATCHA (training-split users only).

For each disorder D we compare word frequencies in D's posts (homepage + Super-Topic) with
those in clean-control homepage posts, using the weighted log-odds ratio with an
informative Dirichlet prior (Monroe, Colaresi & Quinn, 2008). The prior is the pooled
frequency over all cohorts. Words are ranked by z-score.

Privacy: a word is kept only if at least ``min_users`` distinct users wrote it
(k-anonymity), so no rare string that could identify someone ever reaches the LLM filter.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..data.matcha import CONTROL, DISORDERS
from ..text import is_all_cjk

STOPWORDS = set(
    """
    的 了 是 我 你 他 她 它 们 我们 你们 他们 她们 这 那 这个 那个 这些 那些 这样 那样 一个 一些 什么 怎么 怎么样
    为什么 因为 所以 但是 可是 而且 或者 还是 就是 不是 没有 有 在 和 与 跟 对 把 被 让 给 从 到 向 于 以 及
    也 都 就 才 又 还 再 很 太 最 更 非常 特别 真的 真是 其实 然后 现在 今天 明天 昨天 时候 已经 一直 一下 一点
    可以 可能 应该 需要 觉得 知道 看到 感觉 自己 大家 别人 东西 事情 吗 呢 吧 啊 呀 哦 嗯 哈 哈哈 哈哈哈 啦 嘛
    哇 诶 欸 么 着 过 得 地 之 其 此 该 各 每 某 等 如果 虽然 只是 只有 不过 而 但 则 却 并 且 即 如 若 因 为
    个 只 次 些 点 下 上 中 里 外 前 后 们 来 去 说 想 要 会 能 看 做 好 多 少 大 小 年 月 日 号 天 人
    微博 视频 图片 链接 网页 转发 评论 点赞 分享 关注 超话 全文 展开 收起
    """.split()
)
_LATIN = re.compile(r"^[a-z]{2,12}$")


def is_candidate(word: str) -> bool:
    w = word.strip().lower()
    if w in STOPWORDS or len(w) < 2 or len(w) > 8:
        return False
    return is_all_cjk(w) or bool(_LATIN.match(w))


def count_cohort_words(posts: pd.DataFrame, segment) -> tuple[dict[str, Counter], dict[str, Counter]]:
    """Return (token counts, distinct-user counts) per cohort, segmenting each post once."""
    tokens: dict[str, Counter] = defaultdict(Counter)
    users: dict[str, Counter] = defaultdict(Counter)
    note_cohorts = posts.groupby("note_id")["cohort"].agg(lambda s: sorted(set(s)))
    unique = posts.drop_duplicates("note_id")[["user_id", "note_id", "text"]]
    for user_id, group in tqdm(unique.groupby("user_id", sort=False), desc="segmenting users", mininterval=5):
        per_cohort_words: dict[str, set] = defaultdict(set)
        for note_id, text in zip(group["note_id"], group["text"]):
            words = [w.lower() for w in segment(text) if is_candidate(w)]
            for c in note_cohorts[note_id]:
                tokens[c].update(words)
                per_cohort_words[c].update(words)
        for c, ws in per_cohort_words.items():
            users[c].update(ws)
    return tokens, users


def log_odds_dirichlet(target: Counter, background: Counter, prior: Counter, alpha0: float = 1000.0) -> pd.DataFrame:
    vocab = sorted(set(target) | set(background))
    y_t = np.array([target.get(w, 0) for w in vocab], dtype=np.float64)
    y_b = np.array([background.get(w, 0) for w in vocab], dtype=np.float64)
    p = np.array([prior.get(w, 0) for w in vocab], dtype=np.float64)
    a = alpha0 * (p + 0.01) / (p.sum() + 0.01 * len(vocab))
    a0 = a.sum()
    n_t, n_b = y_t.sum(), y_b.sum()
    delta = np.log((y_t + a) / (n_t + a0 - y_t - a)) - np.log((y_b + a) / (n_b + a0 - y_b - a))
    var = 1.0 / (y_t + a) + 1.0 / (y_b + a)
    return pd.DataFrame({"word": vocab, "delta": delta, "z": delta / np.sqrt(var), "count_target": y_t, "count_control": y_b})


def mine_candidates(
    posts: pd.DataFrame,
    segment,
    *,
    disorders: Iterable[str] = DISORDERS,
    split: str = "train",
    alpha0: float = 1000.0,
    z_min: float = 3.0,
    min_users: int = 5,
    top_k: int = 1500,
) -> pd.DataFrame:
    train = posts[posts["split"] == split]
    tokens, users = count_cohort_words(train, segment)
    prior = Counter()
    for c in tokens.values():
        prior.update(c)
    frames = []
    for d in disorders:
        if d not in tokens:
            continue
        scores = log_odds_dirichlet(tokens[d], tokens[CONTROL], prior, alpha0)
        scores["users_target"] = scores["word"].map(users[d]).fillna(0).astype(int)
        scores["users_control"] = scores["word"].map(users[CONTROL]).fillna(0).astype(int)
        keep = scores[(scores["z"] >= z_min) & (scores["users_target"] >= min_users)]
        keep = keep.sort_values("z", ascending=False).head(top_k).assign(disorder=d)
        frames.append(keep)
    cols = ["disorder", "word", "z", "delta", "count_target", "count_control", "users_target", "users_control"]
    return pd.concat(frames, ignore_index=True)[cols]

"""Filtering mined candidate words with DeepSeek (OpenAI-compatible chat API).

Only *word lists* are sent to the API, never posts or user data. Candidates have already
passed a k-anonymity filter (>= min_users distinct authors) in ``mine.py``.

Each (disorder, word) pair is judged ``votes`` times at a non-zero temperature. A word is kept
on a majority vote, and the agreement rate is recorded so the lexicon's stability can be
reported. All responses are cached in a JSONL file, so interrupted runs resume for free and
re-runs cost nothing. Batches are sent concurrently (``max_workers``).

The API key is read from $DEEPSEEK_API_KEY, or from a git-ignored ``.env`` file at the repo root
(one line: ``DEEPSEEK_API_KEY=sk-...``), so it never appears in commands or tracked files.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from ..config import REPO_ROOT
from ..data.matcha import DISORDER_NAMES
from ..utils import get_logger

PROMPT_VERSION = "v1"
CATEGORIES = ("emotion", "cognition", "somatic", "behavior", "clinical", "community", "social")
log = get_logger()

SYSTEM_PROMPT = (
    "You are a clinical psychologist and a Chinese computational linguist. You are curating a domain "
    "lexicon that will guide masked-language-model pre-training on Chinese Weibo mental-health text. "
    "You answer with strict JSON only."
)

USER_TEMPLATE = """Target condition: {name_en} ({name_zh}).

The {n} candidate words below are statistically over-represented in Weibo posts written by members of the {name_zh} community, compared with ordinary users. Judge each word by its meaning in Chinese social-media usage.

Keep a word (keep = 1) only if it is clearly related to mental health or to the lived experience of {name_en}, in one of these categories:
- emotion: affective states (e.g. 绝望, 烦躁, 崩溃)
- cognition: thoughts, beliefs, self-evaluation (e.g. 自责, 胡思乱想, 没用)
- somatic: bodily symptoms (e.g. 失眠, 心悸, 头晕, 胸闷)
- behavior: actions and behavioural symptoms (e.g. 自残, 反复洗手, 拖延, 暴食)
- clinical: diagnosis, treatment, medication, services (e.g. 确诊, 复诊, 住院, 副作用)
- community: Weibo mental-health community slang (e.g. 郁友, 焦友, emo)
- social: relational or life stressors closely tied to the condition (e.g. 孤立, 霸凌, 冷暴力)

Reject (keep = 0): function words, everyday topics (food, celebrities, shopping, games), named entities (people, brands, places, hospitals), advertising, platform boilerplate, and anything that could identify a person.

Return JSON only, in this exact shape, with one entry per input word in the same order:
{{"items": [{{"word": "<word>", "keep": 0 or 1, "category": "<one of: {categories} or none>"}}]}}

Words:
{words}"""


def _key_from_dotenv(name: str = "DEEPSEEK_API_KEY") -> str | None:
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return None
    for line in env_file.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key.strip() == name:
            return value.strip().strip('"').strip("'")
    return None


class DeepSeekClient:
    def __init__(
        self,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        api_key: str | None = None,
        temperature: float = 0.7,
        timeout: int = 120,
        max_retries: int = 5,
    ):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY") or _key_from_dotenv()
        if not self.api_key:
            raise RuntimeError("Set DEEPSEEK_API_KEY, or put DEEPSEEK_API_KEY=sk-... in the repo's .env file.")
        self.model, self.base_url = model, base_url.rstrip("/")
        self.temperature, self.timeout, self.max_retries = temperature, timeout, max_retries
        self.usage = Counter()
        self._lock = threading.Lock()

    def chat_json(self, system: str, user: str) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        for attempt in range(self.max_retries):
            try:
                r = requests.post(f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"retryable status {r.status_code}")
                r.raise_for_status()
                payload = r.json()
                usage = payload.get("usage", {})
                with self._lock:
                    self.usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    self.usage["completion_tokens"] += usage.get("completion_tokens", 0)
                    self.usage["calls"] += 1
                return json.loads(payload["choices"][0]["message"]["content"])
            except (requests.RequestException, json.JSONDecodeError, KeyError) as exc:
                wait = 2 ** attempt
                log.warning("DeepSeek call failed (%s); retry %d/%d in %ds", exc, attempt + 1, self.max_retries, wait)
                time.sleep(wait)
        raise RuntimeError("DeepSeek call failed after retries")


def build_prompt(disorder: str, words: list[str]) -> str:
    name_en, name_zh = DISORDER_NAMES[disorder]
    return USER_TEMPLATE.format(
        name_en=name_en, name_zh=name_zh, n=len(words), categories=", ".join(CATEGORIES), words="\n".join(words)
    )


def parse_items(response: dict, words: list[str]) -> dict[str, tuple[int, str]]:
    out = {}
    for item in response.get("items", []):
        word = str(item.get("word", "")).strip()
        if word in words:
            keep = 1 if str(item.get("keep", 0)).strip() in ("1", "true", "True") else 0
            cat = str(item.get("category", "none")).strip().lower()
            out[word] = (keep, cat if cat in CATEGORIES else "none")
    return out


def estimate_cost(n_words: int, votes: int, batch_size: int, price_in: float, price_out: float) -> dict:
    calls = votes * -(-n_words // batch_size)
    prompt_tokens = calls * 650 + votes * n_words * 3
    completion_tokens = votes * n_words * 22
    usd = prompt_tokens / 1e6 * price_in + completion_tokens / 1e6 * price_out
    return {"calls": calls, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "usd": round(usd, 4)}


def _load_cache(path: Path) -> dict:
    cache = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                cache[(r["model"], r["prompt_version"], r["disorder"], r["word"], r["vote"])] = (r["keep"], r["category"])
    return cache


def filter_with_deepseek(
    candidates: pd.DataFrame,
    client: DeepSeekClient,
    cache_path: Path,
    *,
    votes: int = 3,
    batch_size: int = 100,
    max_workers: int = 8,
) -> pd.DataFrame:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = _load_cache(cache_path)
    model = client.model

    jobs = []
    for disorder, group in candidates.groupby("disorder"):
        words = group["word"].tolist()
        for vote in range(votes):
            pending = [w for w in words if (model, PROMPT_VERSION, disorder, w, vote) not in cache]
            jobs += [(disorder, vote, pending[i : i + batch_size]) for i in range(0, len(pending), batch_size)]
    log.info("%d API calls to make (%d workers)", len(jobs), max_workers)

    def run(job):
        disorder, vote, batch = job
        return job, parse_items(client.chat_json(SYSTEM_PROMPT, build_prompt(disorder, batch)), batch)

    with open(cache_path, "a", encoding="utf-8") as fout, ThreadPoolExecutor(max_workers) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            (disorder, vote, batch), parsed = future.result()
            for w in batch:
                keep, cat = parsed.get(w, (0, "missing"))
                cache[(model, PROMPT_VERSION, disorder, w, vote)] = (keep, cat)
                fout.write(json.dumps({
                    "model": model, "prompt_version": PROMPT_VERSION, "disorder": disorder,
                    "word": w, "vote": vote, "keep": keep, "category": cat,
                }, ensure_ascii=False) + "\n")
            fout.flush()
            if done % 20 == 0 or done == len(jobs):
                log.info("%d/%d calls done", done, len(jobs))

    rows = []
    for disorder, word in zip(candidates["disorder"], candidates["word"]):
        judged = [cache[(model, PROMPT_VERSION, disorder, word, v)] for v in range(votes)]
        keeps = sum(k for k, _ in judged)
        cats = Counter(c for k, c in judged if k == 1)
        rows.append({
            "disorder": disorder, "word": word, "keep_votes": keeps, "votes": votes,
            "keep": int(keeps * 2 > votes),
            "category": cats.most_common(1)[0][0] if cats else "none",
            "agreement": max(keeps, votes - keeps) / votes,
        })
    annotations = candidates.merge(pd.DataFrame(rows), on=["disorder", "word"], how="left")
    log.info("DeepSeek usage: %s", dict(client.usage))
    return annotations

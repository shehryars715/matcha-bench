"""DAPT corpus: segmentation, tokenisation and packing into 128-token chunks, done once.

Every pre-training variant (random, depression lexicon, DeepSeek lexicon) reads the **same**
cache, so the only thing that differs between runs is which words are masked. To make that
possible:

* jieba is primed with the *union* of all lexicons, so lexicon words are always segmented as
  whole words, identically for every variant;
* each token stores the id of the word it belongs to (``word_ids``) and whether it starts a new
  word occurrence (``word_start``). At training time a lexicon becomes a boolean lookup over
  the word vocabulary, so swapping lexicons needs no re-tokenisation.

Following Chinese MentalBERT, cleaned posts are concatenated and cut into chunks of
``max_seq_length - 2`` tokens (+[CLS]/[SEP]). Chunks are cut only at word boundaries.
"""

from __future__ import annotations

import json
from array import array
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm

_WORKER: dict = {}


def _init_worker(tokenizer_name: str, userdict_words: list[str]) -> None:
    import jieba
    from transformers import AutoTokenizer

    jieba.setLogLevel(60)
    for w in userdict_words:
        jieba.add_word(w)
    _WORKER["tokenizer"] = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    _WORKER["jieba"] = jieba


def encode_texts(texts: list[str]) -> list[list[tuple[str, list[int]]]]:
    """For each text return its word occurrences as (word, token ids)."""
    tokenizer, jieba = _WORKER["tokenizer"], _WORKER["jieba"]
    enc = tokenizer(texts, add_special_tokens=False, return_offsets_mapping=True)
    out = []
    for text, ids, offsets in zip(texts, enc["input_ids"], enc["offset_mapping"]):
        spans = list(jieba.tokenize(text))
        char_word = np.zeros(max(len(text), 1), dtype=np.int64)
        for k, (_, start, end) in enumerate(spans):
            char_word[start:end] = k
        words: list[tuple[str, list[int]]] = []
        prev = -1
        for tid, (start, _) in zip(ids, offsets):
            k = int(char_word[start]) if start < len(text) else prev
            if k != prev or not words:
                words.append((spans[k][0].lower() if k >= 0 else "", [tid]))
                prev = k
            else:
                words[-1][1].append(tid)
        out.append(words)
    return out


def _batched(texts: Iterable[str], size: int) -> Iterator[list[str]]:
    batch = []
    for t in texts:
        batch.append(t)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_corpus_cache(
    texts: list[str],
    out_file: Path,
    *,
    tokenizer_name: str,
    word2id: dict[str, int],
    max_seq_length: int = 128,
    userdict_words: Iterable[str] = (),
    num_workers: int = 0,
    batch_size: int = 512,
) -> dict:
    """Pack ``texts`` into chunks and write ``out_file`` (.npz). ``word2id`` is updated in place."""
    limit = max_seq_length - 2
    ids, wids, wstart = array("i"), array("i"), array("b")
    offsets = [0]
    cur_ids: list[int] = []
    cur_wids: list[int] = []
    cur_ws: list[int] = []
    skipped_long = 0

    def flush():
        if cur_ids:
            ids.extend(cur_ids)
            wids.extend(cur_wids)
            wstart.extend(cur_ws)
            offsets.append(len(ids))
            cur_ids.clear(), cur_wids.clear(), cur_ws.clear()

    userdict_words = sorted(set(userdict_words))
    batches = _batched(texts, batch_size)
    total = -(-len(texts) // batch_size)
    if num_workers and num_workers > 1:
        pool = Pool(num_workers, initializer=_init_worker, initargs=(tokenizer_name, userdict_words))
        results = pool.imap(encode_texts, batches, chunksize=1)
    else:
        pool = None
        _init_worker(tokenizer_name, userdict_words)
        results = map(encode_texts, batches)

    try:
        for batch in tqdm(results, total=total, desc=f"packing {Path(out_file).name}", mininterval=10):
            for words in batch:
                for word, toks in words:
                    if len(toks) > limit:
                        skipped_long += 1
                        continue
                    if len(cur_ids) + len(toks) > limit:
                        flush()
                    wid = word2id.setdefault(word, len(word2id))
                    cur_ids.extend(toks)
                    cur_wids.extend([wid] * len(toks))
                    cur_ws.append(1)
                    cur_ws.extend([0] * (len(toks) - 1))
        flush()
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_file,
        input_ids=np.frombuffer(ids, dtype=np.int32),
        word_ids=np.frombuffer(wids, dtype=np.int32),
        word_start=np.frombuffer(wstart, dtype=np.int8),
        offsets=np.asarray(offsets, dtype=np.int64),
    )
    return {"texts": len(texts), "chunks": len(offsets) - 1, "tokens": len(ids), "skipped_long_words": skipped_long}


def save_word_vocab(word2id: dict[str, int], path: Path) -> None:
    words = [""] * len(word2id)
    for w, i in word2id.items():
        words[i] = w
    with open(path, "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False)


def load_word_vocab(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def lexicon_lookup(words: list[str], lexicon: set[str] | None) -> np.ndarray | None:
    if not lexicon:
        return None
    return np.fromiter((w in lexicon for w in words), dtype=bool, count=len(words))


class ChunkDataset(Dataset):
    """Serves packed chunks with [CLS]/[SEP] plus per-token word-start and lexicon flags."""

    def __init__(self, npz_path: Path, cls_id: int, sep_id: int, lookup: np.ndarray | None = None, max_chunks: int | None = None):
        data = np.load(npz_path)
        self.input_ids = data["input_ids"]
        self.word_ids = data["word_ids"]
        self.word_start = data["word_start"]
        self.offsets = data["offsets"]
        self.cls_id, self.sep_id, self.lookup = cls_id, sep_id, lookup
        n = len(self.offsets) - 1
        self.n = min(n, max_chunks) if max_chunks else n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        s, e = int(self.offsets[i]), int(self.offsets[i + 1])
        lex = self.lookup[self.word_ids[s:e]].astype(np.int8).tolist() if self.lookup is not None else [0] * (e - s)
        return {
            "input_ids": [self.cls_id] + self.input_ids[s:e].tolist() + [self.sep_id],
            "word_start": [1] + self.word_start[s:e].tolist() + [1],
            "lexicon": [0] + lex + [0],
        }

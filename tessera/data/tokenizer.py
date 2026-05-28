"""A byte-level BPE tokenizer (minbpe-style), trained from a corpus.

Byte-level => every string is encodable (no UNK, no unicode holes); BPE merges the most
frequent adjacent pair repeatedly to build a vocabulary that compresses common substrings.
This is the real tokenizer used by the text and multimodal pipelines.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class ByteBPETokenizer:
    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special: dict[str, int] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special)

    def train(self, texts: list[str], vocab_size: int, min_freq: int = 2) -> ByteBPETokenizer:
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256 (the byte alphabet)")
        n_merges = vocab_size - 256
        seqs = [list(t.encode("utf-8")) for t in texts]
        for i in range(n_merges):
            counts: Counter[tuple[int, int]] = Counter()
            for s in seqs:
                counts.update(zip(s, s[1:], strict=False))
            if not counts:
                break
            pair, freq = counts.most_common(1)[0]
            if freq < min_freq:
                break
            new_id = 256 + i
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            seqs = [_merge(s, pair, new_id) for s in seqs]
        return self

    def add_special(self, name: str) -> int:
        if name not in self.special:
            self.special[name] = len(self.vocab) + len(self.special)
        return self.special[name]

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:], strict=False))
            # merge the pair learned earliest (lowest new_id) that is present
            pair = min(pairs, key=lambda p: self.merges.get(p, 1 << 30))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        rev_special = {v: k for k, v in self.special.items()}
        parts: list[bytes] = []
        for i in ids:
            if i in self.vocab:
                parts.append(self.vocab[i])
            elif i in rev_special:
                parts.append(rev_special[i].encode("utf-8"))
        return b"".join(parts).decode("utf-8", errors="replace")

    # -- persistence ----------------------------------------------------------
    def save(self, path: str | Path) -> None:
        data = {
            "merges": [[list(k), v] for k, v in self.merges.items()],
            "special": self.special,
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> ByteBPETokenizer:
        data = json.loads(Path(path).read_text())
        tok = cls()
        for (a, b), new_id in data["merges"]:
            tok.merges[(a, b)] = new_id
            tok.vocab[new_id] = tok.vocab[a] + tok.vocab[b]
        tok.special = data.get("special", {})
        return tok

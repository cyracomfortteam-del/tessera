"""Text packing + streaming for LM / distillation training.

Documents are concatenated with an EOS separator and chopped into fixed-length windows; each
window yields (input, target) where target is input shifted by one. `stream_windows` does the
same over an unbounded token iterator with O(seq_len) memory — the shape real pretraining
pipelines need.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch
from torch.utils.data import Dataset


class PackedTextDataset(Dataset):
    def __init__(self, token_lists: list[list[int]], seq_len: int, eos_id: int):
        stream: list[int] = []
        for toks in token_lists:
            stream.extend(toks)
            stream.append(eos_id)
        self.data = torch.tensor(stream, dtype=torch.long)
        self.seq_len = seq_len
        self.n = max(0, (len(self.data) - 1) // seq_len)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = idx * self.seq_len
        chunk = self.data[s : s + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


def stream_windows(
    tokens: Iterable[int], seq_len: int, eos_id: int | None = None
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield (input, target) windows from a token stream using a rolling buffer."""
    buf: list[int] = []
    for tok in tokens:
        buf.append(tok)
        if len(buf) == seq_len + 1:
            t = torch.tensor(buf, dtype=torch.long)
            yield t[:-1], t[1:]
            buf = []
    if eos_id is not None and len(buf) > 1:
        buf += [eos_id] * (seq_len + 1 - len(buf))
        t = torch.tensor(buf[: seq_len + 1], dtype=torch.long)
        yield t[:-1], t[1:]

"""InferenceEngine — drives the model under the continuous-batching scheduler.

The scheduler decides *which* sequences run and owns the KV-block budget; the engine does
the actual model forwards. Each running sequence keeps its own incremental KV cache for
compute, while the shared BlockAllocator models the real memory pressure that governs
admission (so OOM shows up as back-pressure on the waiting queue, exactly as in a real
serving stack).
"""

from __future__ import annotations

import torch

from tessera.model.transformer import Transformer
from tessera.serve.api import Request, RequestOutput
from tessera.serve.paged_kv import BlockAllocator
from tessera.serve.scheduler import Scheduler, Sequence


class InferenceEngine:
    def __init__(
        self,
        model: Transformer,
        max_batch: int = 8,
        block_size: int = 16,
        num_blocks: int = 4096,
        kv_dtype: torch.dtype | None = None,
    ):
        self.model = model.eval()
        self.allocator = BlockAllocator(num_blocks)
        self.scheduler = Scheduler(self.allocator, block_size, max_running=max_batch)
        self.kv_dtype = kv_dtype or model.tok_emb.weight.dtype
        self._cache: dict[str, object] = {}
        self._pos: dict[str, int] = {}

    def add_request(self, request: Request) -> None:
        self.scheduler.add(request)

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    @torch.no_grad()
    def _start(self, seq: Sequence) -> None:
        rid = seq.request.request_id
        cache = self.model.new_cache(1, dtype=self.kv_dtype)
        x = torch.tensor([seq.request.prompt_tokens], device=self.model.device, dtype=torch.long)
        logits = self.model(x, cache=cache, start_pos=0)[:, -1]
        self._cache[rid] = cache
        self._pos[rid] = seq.prompt_len
        self._emit(seq, logits)

    @torch.no_grad()
    def _decode(self, seq: Sequence) -> None:
        rid = seq.request.request_id
        x = torch.tensor([[seq.last_token]], device=self.model.device, dtype=torch.long)
        logits = self.model(x, cache=self._cache[rid], start_pos=self._pos[rid])[:, -1]
        self._pos[rid] += 1
        self._emit(seq, logits)

    def _emit(self, seq: Sequence, logits: torch.Tensor) -> None:
        # Block management (allocation / preemption) is owned by Scheduler.schedule();
        # here we only advance the token stream.
        p = seq.request.params
        token = int(Transformer._sample(logits, p.temperature, p.top_k).item())
        seq.append(token)

    def step(self) -> list[RequestOutput]:
        running = self.scheduler.schedule()
        finished: list[RequestOutput] = []
        for seq in list(running):
            rid = seq.request.request_id
            if rid not in self._cache:
                self._start(seq)
            else:
                self._decode(seq)
            if seq.is_finished():
                finished.append(self._collect(seq, finished=True))
                self.scheduler.finish(seq)
                self._cache.pop(rid, None)
                self._pos.pop(rid, None)
        return finished

    def _collect(self, seq: Sequence, finished: bool) -> RequestOutput:
        return RequestOutput(
            request_id=seq.request.request_id,
            prompt_tokens=list(seq.request.prompt_tokens),
            output_tokens=seq.tokens[seq.prompt_len :],
            finished=finished,
        )

    def run(self, requests: list[Request]) -> dict[str, RequestOutput]:
        """Convenience: enqueue everything and step until the world is drained."""
        for r in requests:
            self.add_request(r)
        results: dict[str, RequestOutput] = {}
        while self.has_unfinished():
            for out in self.step():
                results[out.request_id] = out
        return results

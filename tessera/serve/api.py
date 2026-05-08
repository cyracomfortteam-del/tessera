"""Request/response types shared by the scheduler, engine, and Rust gateway."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SamplingParams:
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: int | None = None
    eos_id: int | None = None


@dataclass
class Request:
    request_id: str
    prompt_tokens: list[int]
    params: SamplingParams = field(default_factory=SamplingParams)


@dataclass
class RequestOutput:
    request_id: str
    prompt_tokens: list[int]
    output_tokens: list[int]
    finished: bool = False

    @property
    def all_tokens(self) -> list[int]:
        return self.prompt_tokens + self.output_tokens

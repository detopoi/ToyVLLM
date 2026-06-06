from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from toyvllm.sampling import SamplingParams


class SequenceStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


class FinishReason(str, Enum):
    EOS = "eos"
    LENGTH = "length"


@dataclass
class Sequence:
    """一条生成请求在调度器中的完整生命周期状态。"""

    request_id: int
    prompt_token_ids: list[int]
    max_new_tokens: int
    eos_token_ids: frozenset[int]
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    status: SequenceStatus = SequenceStatus.WAITING
    output_token_ids: list[int] = field(default_factory=list)
    finish_reason: FinishReason | None = None
    admitted_step: int | None = None
    finished_step: int | None = None

    def __post_init__(self) -> None:
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids 不能为空")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens 必须大于 0")

    @property
    def last_token_id(self) -> int:
        if not self.output_token_ids:
            raise RuntimeError("请求尚未生成第一个 token")
        return self.output_token_ids[-1]

    def append_token(self, token_id: int) -> FinishReason | None:
        if self.status is not SequenceStatus.RUNNING:
            raise RuntimeError("只有 RUNNING 请求可以追加 token")

        self.output_token_ids.append(token_id)
        if token_id in self.eos_token_ids:
            return FinishReason.EOS
        if len(self.output_token_ids) >= self.max_new_tokens:
            return FinishReason.LENGTH
        return None


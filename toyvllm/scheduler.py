from __future__ import annotations

from collections import deque

from toyvllm.sampling import SamplingParams
from toyvllm.sequence import FinishReason, Sequence, SequenceStatus


class Scheduler:
    """固定并发上限、FIFO 等待队列的最小连续批处理调度器。

    Scheduler 只回答“哪些请求本轮运行”，不执行模型，也不持有 KV Cache。
    """

    def __init__(self, max_num_seqs: int) -> None:
        if max_num_seqs <= 0:
            raise ValueError("max_num_seqs 必须大于 0")
        self.max_num_seqs = max_num_seqs
        self._next_request_id = 0
        self._waiting: deque[Sequence] = deque()
        self._running: dict[int, Sequence] = {}
        self._finished: dict[int, Sequence] = {}
        self._completion_order: list[int] = []

    def add_request(
        self,
        prompt_token_ids: list[int],
        *,
        max_new_tokens: int,
        eos_token_ids: set[int],
        sampling_params: SamplingParams | None = None,
    ) -> Sequence:
        sequence = Sequence(
            request_id=self._next_request_id,
            prompt_token_ids=list(prompt_token_ids),
            max_new_tokens=max_new_tokens,
            eos_token_ids=frozenset(eos_token_ids),
            sampling_params=sampling_params or SamplingParams(),
        )
        self._next_request_id += 1
        self._waiting.append(sequence)
        return sequence

    @property
    def waiting(self) -> tuple[Sequence, ...]:
        return tuple(self._waiting)

    @property
    def running(self) -> tuple[Sequence, ...]:
        # dict 保留插入顺序，运行 batch 因而保持稳定，方便复现和调试。
        return tuple(self._running.values())

    @property
    def finished(self) -> tuple[Sequence, ...]:
        return tuple(self._finished.values())

    @property
    def completion_order(self) -> tuple[int, ...]:
        return tuple(self._completion_order)

    @property
    def is_done(self) -> bool:
        return not self._waiting and not self._running

    def admit_waiting(self, *, step: int) -> tuple[Sequence, ...]:
        """按 FIFO 顺序用等待请求填满空闲运行槽位。"""

        admitted: list[Sequence] = []
        while self._waiting and len(self._running) < self.max_num_seqs:
            sequence = self._waiting.popleft()
            sequence.status = SequenceStatus.RUNNING
            sequence.admitted_step = step
            self._running[sequence.request_id] = sequence
            admitted.append(sequence)
        return tuple(admitted)

    def append_token(
        self,
        sequence: Sequence,
        token_id: int,
        *,
        step: int,
    ) -> FinishReason | None:
        if sequence.request_id not in self._running:
            raise RuntimeError("请求不在 RUNNING 集合中")

        finish_reason = sequence.append_token(token_id)
        if finish_reason is not None:
            self._finish(sequence, finish_reason, step=step)
        return finish_reason

    def _finish(
        self,
        sequence: Sequence,
        reason: FinishReason,
        *,
        step: int,
    ) -> None:
        del self._running[sequence.request_id]
        sequence.status = SequenceStatus.FINISHED
        sequence.finish_reason = reason
        sequence.finished_step = step
        self._finished[sequence.request_id] = sequence
        self._completion_order.append(sequence.request_id)


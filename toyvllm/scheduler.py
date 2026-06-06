from __future__ import annotations

"""请求调度器：只管理状态和运行槽位，不执行网络。"""

from collections import deque

from toyvllm.sampling import SamplingParams
from toyvllm.sequence import FinishReason, Sequence, SequenceStatus


class Scheduler:
    """固定并发上限、FIFO 等待队列的最小连续批处理调度器。

    Scheduler 只回答“哪些请求本轮运行”，不执行模型，也不持有 KV Cache。

    这里刻意把调度策略与模型执行分开：

    - Scheduler 是控制面，管理 WAITING/RUNNING/FINISHED；
    - Engine 是数据面，把 RUNNING 请求打包成 Tensor 并调用 GPU。

    以后增加 token budget、优先级或 KV Block 容量时，主要修改 Scheduler；
    以后更换 Attention 内核或缓存布局时，主要修改 Engine。
    """

    def __init__(self, max_num_seqs: int) -> None:
        if max_num_seqs <= 0:
            raise ValueError("max_num_seqs 必须大于 0")
        self.max_num_seqs = max_num_seqs
        self._next_request_id = 0

        # waiting 使用 deque，popleft() 可以 O(1) 实现 FIFO。FIFO 虽然简单，
        # 但至少保证先到请求不会被后来请求无限插队。
        self._waiting: deque[Sequence] = deque()

        # running/finished 用 request_id 建索引。Python dict 保留插入顺序，
        # 因此每轮 batch 行顺序稳定，有利于采样复现和调度轨迹排查。
        self._running: dict[int, Sequence] = {}
        self._finished: dict[int, Sequence] = {}

        # finished 字典按 request_id 查询；completion_order 单独记录实际完成先后。
        self._completion_order: list[int] = []

    def add_request(
        self,
        prompt_token_ids: list[int],
        *,
        max_new_tokens: int,
        eos_token_ids: set[int],
        sampling_params: SamplingParams | None = None,
    ) -> Sequence:
        """创建 WAITING 请求并放到 FIFO 队尾。

        add_request 不会立即让请求运行。是否有资源接纳它，只能由 admit_waiting
        根据当前空槽决定。这个区分使请求可以在 Engine 运行期间随时到达。
        """

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
        # finished 中保留的是结果，不属于待执行工作，因此不参与 done 判断。
        return not self._waiting and not self._running

    def admit_waiting(self, *, step: int) -> tuple[Sequence, ...]:
        """按 FIFO 顺序用等待请求填满空闲运行槽位。

        返回值只包含“本轮刚刚接纳”的请求。Engine 会对它们执行 Prefill；
        已经在 running 中的旧请求则走 Decode，二者不能混淆。
        """

        admitted: list[Sequence] = []
        while self._waiting and len(self._running) < self.max_num_seqs:
            sequence = self._waiting.popleft()

            # 状态修改和运行集合插入放在同一个方法中，维持不变量：
            # status=RUNNING 当且仅当 request_id 位于 _running。
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
        """提交 Engine 本轮生成的 token，并在必要时完成请求。"""

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
        """原子地完成状态迁移并从运行槽位中移除请求。

        del _running 发生后，len(_running) 立即减一。同一调度轮次后续调用
        admit_waiting 时就能看到空槽，不必等待整个旧 batch 全部结束。
        """

        del self._running[sequence.request_id]
        sequence.status = SequenceStatus.FINISHED
        sequence.finish_reason = reason
        sequence.finished_step = step
        self._finished[sequence.request_id] = sequence
        self._completion_order.append(sequence.request_id)

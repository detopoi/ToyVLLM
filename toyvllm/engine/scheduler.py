from __future__ import annotations

"""请求调度器：只管理状态和运行槽位，不执行网络。"""

from collections import deque
from dataclasses import dataclass

from toyvllm.engine.block_manager import (
    BlockManager,
    BlockTable,
    OutOfBlocksError,
    PhysicalTokenSlot,
)
from toyvllm.engine.sequence import FinishReason, Sequence, SequenceStatus
from toyvllm.sampling import SamplingParams


@dataclass(frozen=True)
class ScheduledPrefill:
    """Scheduler 为一次 Prompt chunk 固定下来的控制面计划。"""

    sequence: Sequence
    history_table: BlockTable
    slots: tuple[PhysicalTokenSlot, ...]

    @property
    def num_tokens(self) -> int:
        return len(self.slots)


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
    def prefilling(self) -> tuple[Sequence, ...]:
        """已经占用运行槽，但 Prompt 尚未全部写入 KV Cache 的请求。"""

        return tuple(
            sequence
            for sequence in self._running.values()
            if not sequence.is_prefill_complete
        )

    @property
    def decoding(self) -> tuple[Sequence, ...]:
        """已经产生首 token、下一轮可以执行单 token Decode 的请求。"""

        return tuple(
            sequence
            for sequence in self._running.values()
            if sequence.is_prefill_complete
        )

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

    def admit_waiting(
        self,
        *,
        step: int,
        max_sequences: int | None = None,
    ) -> tuple[Sequence, ...]:
        """按 FIFO 顺序用等待请求填满空闲运行槽位。

        返回值只包含“本轮刚刚接纳”的请求。Engine 会对它们执行 Prefill；
        已经在 running 中的旧请求则走 Decode，二者不能混淆。
        """

        if max_sequences is not None and max_sequences <= 0:
            raise ValueError("max_sequences 必须大于 0")

        admitted: list[Sequence] = []
        while (
            self._waiting
            and len(self._running) < self.max_num_seqs
            and (max_sequences is None or len(admitted) < max_sequences)
        ):
            sequence = self._waiting.popleft()

            # 状态修改和运行集合插入放在同一个方法中，维持不变量：
            # status=RUNNING 当且仅当 request_id 位于 _running。
            sequence.status = SequenceStatus.RUNNING
            if sequence.admitted_step is None:
                sequence.admitted_step = step
            sequence.last_admitted_step = step
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
        if not sequence.is_prefill_complete:
            raise RuntimeError("Prompt 尚未完成 Prefill，不能提交生成 token")

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


class PagedScheduler(Scheduler):
    """带 Paged KV 容量管理和 Prefill token budget 的调度器。

    一轮中 Decode 具有优先级：每条 Decode 请求先消耗 1 个 token budget，剩余额度
    再公平分给 Prefill。这里的“混合”指同一调度轮同时包含 Decode 和 Prefill 工作，
    当前 Engine 仍按 Decode -> Prefill 两次模型调用执行，便于先讲清调度语义。
    """

    def __init__(
        self,
        max_num_seqs: int,
        *,
        block_manager: BlockManager,
        max_num_batched_tokens: int,
        max_prefill_chunk_size: int,
        max_mixed_prefill_tokens: int | None,
    ) -> None:
        super().__init__(max_num_seqs)
        if max_num_batched_tokens < max_num_seqs:
            raise ValueError(
                "max_num_batched_tokens 不能小于 max_num_seqs，"
                "否则满载 Decode 无法保证每条请求前进一步"
            )
        if max_prefill_chunk_size <= 0:
            raise ValueError("max_prefill_chunk_size 必须大于 0")
        if (
            max_mixed_prefill_tokens is not None
            and max_mixed_prefill_tokens <= 0
        ):
            raise ValueError("max_mixed_prefill_tokens 必须大于 0")
        self.block_manager = block_manager
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_prefill_chunk_size = max_prefill_chunk_size
        self.max_mixed_prefill_tokens = max_mixed_prefill_tokens
        self._step_preempted_request_ids: list[int] = []

    @property
    def step_preempted_request_ids(self) -> tuple[int, ...]:
        return tuple(self._step_preempted_request_ids)

    def schedule_decode(self, *, step: int) -> tuple[Sequence, ...]:
        """为 Decode 预留增长空间；不足时按 RECOMPUTE 策略抢占请求。"""

        self._step_preempted_request_ids.clear()
        candidates = list(self.decoding)
        while True:
            required_blocks = sum(
                self.block_manager.additional_blocks_required(
                    sequence.request_id,
                    1,
                )
                for sequence in candidates
            )
            if required_blocks <= self.block_manager.stats.num_free_blocks:
                return tuple(candidates)

            # 优先抢占尚未输出首 token 的 Prefill 请求，避免破坏用户正在接收的流。
            victim = self._select_preemption_victim(
                protected_request_ids=set(),
                prefer_prefill=True,
            )
            if victim is None:
                raise OutOfBlocksError("没有可抢占请求，Decode 无法继续增长")
            self._preempt(victim, step=step)
            candidates = [
                sequence
                for sequence in candidates
                if sequence.request_id != victim.request_id
            ]

    def schedule_prefill(
        self,
        *,
        step: int,
        num_decode_tokens: int,
    ) -> tuple[ScheduledPrefill, ...]:
        """接纳请求，并用 Decode 后剩余的 budget 安排本轮 Prompt chunks。"""

        budget = self.max_num_batched_tokens - num_decode_tokens
        if num_decode_tokens and self.max_mixed_prefill_tokens is not None:
            # 当前教学 Engine 的 Decode 和 Prefill 是同轮中的两次模型调用，而不是
            # 一个融合的 token batch。若把全部剩余预算都给 Prefill，Decode 虽然先算，
            # 下一个 token 仍会被很长的 Prefill 调用阻塞。因此混合轮使用更小的上限。
            budget = min(budget, self.max_mixed_prefill_tokens)
        if budget <= 0:
            return ()

        self._admit_waiting_with_blocks(step=step)
        prefilling = self.prefilling
        scheduled: list[ScheduledPrefill] = []
        protected_request_ids: set[int] = set()
        for index, sequence in enumerate(prefilling):
            if budget <= 0:
                break
            if sequence.status is not SequenceStatus.RUNNING:
                continue

            # 动态 fair share 避免第一条长 Prompt 独占整轮 budget。短请求没有用完的
            # 额度会自然留给后面的请求，因此总预算仍可尽量被利用。
            remaining_sequences = len(prefilling) - index
            fair_share = (budget + remaining_sequences - 1) // remaining_sequences
            chunk_size = min(
                sequence.num_prompt_tokens_remaining,
                self.max_prefill_chunk_size,
                fair_share,
            )
            while self.block_manager.max_reservable_tokens(
                sequence.request_id
            ) <= 0:
                victim = self._select_preemption_victim(
                    protected_request_ids=(
                        protected_request_ids | {sequence.request_id}
                    ),
                    prefer_prefill=True,
                )
                if victim is None:
                    break
                self._preempt(victim, step=step)

            chunk_size = min(
                chunk_size,
                self.block_manager.max_reservable_tokens(
                    sequence.request_id
                ),
            )
            if chunk_size <= 0:
                continue

            history_table = self.block_manager.get_block_table(
                sequence.request_id
            )
            slots = self.block_manager.reserve(
                sequence.request_id,
                chunk_size,
            )
            scheduled.append(
                ScheduledPrefill(
                    sequence=sequence,
                    history_table=history_table,
                    slots=slots,
                )
            )
            budget -= chunk_size
            protected_request_ids.add(sequence.request_id)
        return tuple(scheduled)

    def _admit_waiting_with_blocks(self, *, step: int) -> None:
        """按 FIFO 接纳请求，只注册空 BlockTable，后续按 chunk 增量分配。"""

        while self.waiting and len(self.running) < self.max_num_seqs:
            candidate = self.waiting[0]
            required = self.block_manager.blocks_required_for_tokens(
                len(candidate.prompt_token_ids)
                + candidate.max_new_tokens
                - 1
            )
            if required > self.block_manager.num_blocks:
                raise OutOfBlocksError(
                    f"请求 {candidate.request_id} 的最大上下文需要 {required} 个 "
                    f"Block，超过物理池总量 {self.block_manager.num_blocks}；"
                    "该请求即使独占 GPU 也无法完成"
                )

            if candidate.preemption_count:
                recompute_blocks = self.block_manager.blocks_required_for_tokens(
                    candidate.prefill_target_length
                )
                if recompute_blocks > self.block_manager.stats.num_free_blocks:
                    # 被抢占请求如果只拿到一小块空间就立即恢复，很可能在下一轮再次
                    # 被抢占，形成 recompute thrashing。等完整重算工作集可容纳时再重入。
                    break

            sequence = self.admit_waiting(step=step, max_sequences=1)[0]
            self.block_manager.allocate(
                sequence.request_id,
                num_tokens=0,
            )

    def _select_preemption_victim(
        self,
        *,
        protected_request_ids: set[int],
        prefer_prefill: bool,
    ) -> Sequence | None:
        """选择最新进入 RUNNING 的低优先级请求。

        同类请求使用 LIFO：保留更早到达的请求继续前进，减少反复抢占造成的饥饿。
        """

        candidates = [
            sequence
            for sequence in reversed(self.running)
            if sequence.request_id not in protected_request_ids
        ]
        if prefer_prefill:
            for sequence in candidates:
                if not sequence.is_prefill_complete:
                    return sequence
        return candidates[0] if candidates else None

    def _preempt(self, sequence: Sequence, *, step: int) -> None:
        """执行 RECOMPUTE 抢占：释放 KV，保留输出，并放回等待队首。"""

        del self._running[sequence.request_id]
        self.block_manager.free(sequence.request_id)
        sequence.preempt_for_recompute(step=step)
        self._waiting.appendleft(sequence)
        self._step_preempted_request_ids.append(sequence.request_id)

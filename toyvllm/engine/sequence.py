from __future__ import annotations

"""Engine 内部的生成请求数据模型。

这个模块不包含调度策略，也不接触 GPU Tensor。它只描述“一条请求现在走到哪一步”，
是 Scheduler 和 Engine 之间共享的状态对象。
"""

from dataclasses import dataclass, field
from enum import Enum

from toyvllm.sampling import SamplingParams


class SequenceStatus(str, Enum):
    """请求状态；抢占时允许 RUNNING -> WAITING，完成后不再回退。"""

    # 已进入系统，但还没有获得并发槽位，也没有 KV Cache。
    WAITING = "waiting"
    # 已获得并发槽位。Prefill 后拥有 KV Cache，可以继续 Decode。
    RUNNING = "running"
    # 遇到 EOS 或长度上限，后续不再参与模型计算。
    FINISHED = "finished"


class FinishReason(str, Enum):
    """区分请求是自然结束，还是被配置的生成长度截断。"""

    EOS = "eos"
    LENGTH = "length"


@dataclass
class Sequence:
    """一条生成请求在调度器中的完整生命周期状态。

    可以把 Sequence 看作请求的“控制面”数据：

    - prompt/output token 决定模型下一步输入什么；
    - status 决定 Scheduler 是否应该选择它；
    - sampling_params 决定如何从 logits 选择 token；
    - admitted_step/finished_step 用于观察排队和完成时机。

    KV Cache 不放在这里。缓存是体积很大的 GPU 资源，由 Engine 单独管理。这样 Scheduler
    测试时不需要 CUDA，未来把连续 Tensor 换成 Paged KV Cache 也不必改变 Sequence。
    """

    # request_id 在整个 Engine 生命周期内单调递增，是状态表和缓存表的共同索引。
    request_id: int
    # Prompt 始终保持原始紧凑形式，不在 Sequence 中保存 batch padding。
    prompt_token_ids: list[int]
    max_new_tokens: int
    eos_token_ids: frozenset[int]
    sampling_params: SamplingParams = field(default_factory=SamplingParams)

    # 以下字段会随请求执行过程变化。
    status: SequenceStatus = SequenceStatus.WAITING
    output_token_ids: list[int] = field(default_factory=list)
    finish_reason: FinishReason | None = None
    # admitted_step 保留第一次进入 RUNNING 的时刻；last_admitted_step 记录重算重入。
    admitted_step: int | None = None
    last_admitted_step: int | None = None
    finished_step: int | None = None
    # 当前一轮 Prefill/Recompute 已有多少上下文 token 写入 KV Cache。初次 Prefill
    # 只覆盖 Prompt；被抢占后，重算目标还会包含已经生成、但不能丢失的 output token。
    num_prompt_tokens_computed: int = 0
    prefill_target_length: int = field(init=False)
    preemption_count: int = 0
    last_preempted_step: int | None = None
    # 累计记录每次 admission 从 Prefix Cache 恢复的 token，便于 benchmark 观察收益。
    prefix_cache_hit_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids 不能为空")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens 必须大于 0")
        self.prefill_target_length = len(self.prompt_token_ids)

    @property
    def last_token_id(self) -> int:
        """Decode 阶段只需要把上一轮生成的 token 送入模型。"""

        if not self.output_token_ids:
            raise RuntimeError("请求尚未生成第一个 token")
        return self.output_token_ids[-1]

    @property
    def num_prompt_tokens_remaining(self) -> int:
        return self.prefill_target_length - self.num_prompt_tokens_computed

    @property
    def is_prefill_complete(self) -> bool:
        return self.num_prompt_tokens_remaining == 0

    def next_prompt_chunk(self, num_tokens: int) -> list[int]:
        """返回下一段尚未计算的上下文 token，但不提前移动游标。

        初次 Prefill 的目标是 Prompt。Decode 请求被抢占后，目标变为
        ``Prompt + 已生成输出``，这样重算完成时可以从原位置继续生成。
        """

        if num_tokens <= 0:
            raise ValueError("num_tokens 必须大于 0")
        if num_tokens > self.num_prompt_tokens_remaining:
            raise ValueError("Prefill chunk 超过剩余 Prompt 长度")
        start = self.num_prompt_tokens_computed
        context_token_ids = self.prompt_token_ids + self.output_token_ids
        stop = start + num_tokens
        if stop > self.prefill_target_length:
            raise RuntimeError("Prefill chunk 超过当前重算目标")
        return context_token_ids[start:stop]

    def advance_prefill(self, num_tokens: int) -> None:
        """模型和 KV 写回成功后，原子提交本次 Prefill 进度。"""

        if num_tokens <= 0:
            raise ValueError("num_tokens 必须大于 0")
        new_value = self.num_prompt_tokens_computed + num_tokens
        if new_value > self.prefill_target_length:
            raise ValueError("Prefill 进度不能超过当前重算目标")
        self.num_prompt_tokens_computed = new_value

    def restore_cached_prefix(self, num_tokens: int) -> None:
        """在 admission 时恢复整块缓存前缀，不执行模型前向。"""

        if num_tokens < 0:
            raise ValueError("缓存前缀 token 数必须是非负整数")
        if self.num_prompt_tokens_computed != 0:
            raise RuntimeError("只有尚未开始 Prefill 的请求可以恢复缓存")
        if num_tokens >= self.prefill_target_length:
            raise ValueError("Prefix Cache 必须留下至少一个 token 计算 logits")
        self.num_prompt_tokens_computed = num_tokens
        self.prefix_cache_hit_tokens += num_tokens

    def preempt_for_recompute(self, *, step: int) -> None:
        """释放 KV Cache 后退回 WAITING，并准备从头重算完整上下文。"""

        if self.status is not SequenceStatus.RUNNING:
            raise RuntimeError("只有 RUNNING 请求可以被抢占")
        self.status = SequenceStatus.WAITING
        self.num_prompt_tokens_computed = 0
        self.prefill_target_length = (
            len(self.prompt_token_ids) + len(self.output_token_ids)
        )
        self.preemption_count += 1
        self.last_preempted_step = step

    def append_token(self, token_id: int) -> FinishReason | None:
        """记录新 token，并只判断是否满足结束条件。

        这个方法不直接把 status 改成 FINISHED。状态集合由 Scheduler 统一维护，否则
        Sequence 可能已经标记完成，却仍残留在 Scheduler.running 中，形成双重事实来源。
        """

        if self.status is not SequenceStatus.RUNNING:
            raise RuntimeError("只有 RUNNING 请求可以追加 token")

        self.output_token_ids.append(token_id)
        # EOS 优先于长度上限，便于准确报告模型是自然停止还是被截断。
        if token_id in self.eos_token_ids:
            return FinishReason.EOS
        if len(self.output_token_ids) >= self.max_new_tokens:
            return FinishReason.LENGTH
        return None

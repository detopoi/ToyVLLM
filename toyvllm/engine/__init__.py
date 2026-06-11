from __future__ import annotations

"""推理引擎子系统的公共入口。

内部模块按职责拆分，但调用方只需要从 ``toyvllm.engine`` 导入公共类型。
这里使用延迟导入，避免 ``llm_engine -> kv_cache -> block_manager`` 形成循环依赖。
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from toyvllm.engine.block_manager import (
        BlockManager,
        BlockManagerStats,
        BlockTable,
        OutOfBlocksError,
        PhysicalTokenSlot,
    )
    from toyvllm.engine.llm_engine import (
        ContinuousBatchEngine,
        ContinuousBatchResult,
        EngineIteration,
        PagedContinuousBatchEngine,
    )
    from toyvllm.engine.memory_planner import (
        KVCacheCapacityPlan,
        calculate_kv_cache_capacity,
        plan_kv_cache_capacity,
    )
    from toyvllm.engine.scheduler import PagedScheduler, ScheduledPrefill, Scheduler
    from toyvllm.engine.sequence import FinishReason, Sequence, SequenceStatus

__all__ = [
    "BlockManager",
    "BlockManagerStats",
    "BlockTable",
    "ContinuousBatchEngine",
    "ContinuousBatchResult",
    "EngineIteration",
    "FinishReason",
    "KVCacheCapacityPlan",
    "OutOfBlocksError",
    "PagedScheduler",
    "PagedContinuousBatchEngine",
    "PhysicalTokenSlot",
    "ScheduledPrefill",
    "Scheduler",
    "Sequence",
    "SequenceStatus",
    "calculate_kv_cache_capacity",
    "plan_kv_cache_capacity",
]


def __getattr__(name: str) -> Any:
    """按需加载公共对象，让轻量控制面测试不必导入模型和 CUDA 路径。"""

    if name in {
        "ContinuousBatchEngine",
        "ContinuousBatchResult",
        "EngineIteration",
        "PagedContinuousBatchEngine",
    }:
        from toyvllm.engine import llm_engine

        return getattr(llm_engine, name)
    if name in {
        "BlockManager",
        "BlockManagerStats",
        "BlockTable",
        "OutOfBlocksError",
        "PhysicalTokenSlot",
    }:
        from toyvllm.engine import block_manager

        return getattr(block_manager, name)
    if name in {"PagedScheduler", "ScheduledPrefill", "Scheduler"}:
        from toyvllm.engine import scheduler

        return getattr(scheduler, name)
    if name in {
        "KVCacheCapacityPlan",
        "calculate_kv_cache_capacity",
        "plan_kv_cache_capacity",
    }:
        from toyvllm.engine import memory_planner

        return getattr(memory_planner, name)
    if name in {"FinishReason", "Sequence", "SequenceStatus"}:
        from toyvllm.engine import sequence

        return getattr(sequence, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

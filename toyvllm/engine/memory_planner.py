from __future__ import annotations

"""根据 GPU 显存预算自动规划 Paged KV Cache 的物理 Block 数。"""

import gc
from dataclasses import dataclass

import torch

from toyvllm.config import ModelConfig


MIB = 1024**2


@dataclass(frozen=True)
class KVCacheCapacityPlan:
    """一次 KV Cache 容量规划的输入快照和结果。"""

    total_memory_bytes: int
    free_memory_bytes: int
    gpu_memory_utilization: float
    runtime_reserve_bytes: int
    bytes_per_block: int
    workspace_bytes_per_block: int
    available_cache_bytes: int
    num_blocks: int

    @property
    def current_used_bytes(self) -> int:
        return self.total_memory_bytes - self.free_memory_bytes

    @property
    def target_used_bytes(self) -> int:
        return int(self.total_memory_bytes * self.gpu_memory_utilization)

    @property
    def allocated_cache_bytes(self) -> int:
        return self.num_blocks * self.bytes_per_block

    def format(self) -> str:
        return (
            f"利用率目标={self.gpu_memory_utilization:.0%}, "
            f"当前占用={self.current_used_bytes / MIB:.1f} MiB, "
            f"运行余量={self.runtime_reserve_bytes / MIB:.1f} MiB, "
            f"KV预算={self.available_cache_bytes / MIB:.1f} MiB, "
            f"Blocks={self.num_blocks}"
        )


def calculate_kv_cache_capacity(
    *,
    total_memory_bytes: int,
    free_memory_bytes: int,
    gpu_memory_utilization: float,
    runtime_reserve_bytes: int,
    bytes_per_block: int,
    workspace_bytes_per_block: int = 0,
) -> KVCacheCapacityPlan:
    """用显存快照计算可分配 Block 数，不访问 CUDA，便于独立测试。

    预算公式：

        target_used = total_memory * utilization
        cache_budget = target_used - current_used - runtime_reserve

    ``runtime_reserve`` 留给 Prefill/Decode 激活、SDPA workspace 和临时 Tensor。
    利用率未覆盖的显存则是最后一道 OOM 缓冲，两者含义不同。
    """

    if total_memory_bytes <= 0:
        raise ValueError("total_memory_bytes 必须大于 0")
    if not 0 <= free_memory_bytes <= total_memory_bytes:
        raise ValueError("free_memory_bytes 必须位于 [0, total_memory_bytes]")
    if not 0.0 < gpu_memory_utilization <= 1.0:
        raise ValueError("gpu_memory_utilization 必须位于 (0, 1]")
    if runtime_reserve_bytes < 0:
        raise ValueError("runtime_reserve_bytes 不能为负数")
    if bytes_per_block <= 0:
        raise ValueError("bytes_per_block 必须大于 0")
    if workspace_bytes_per_block < 0:
        raise ValueError("workspace_bytes_per_block 不能为负数")

    current_used = total_memory_bytes - free_memory_bytes
    target_used = int(total_memory_bytes * gpu_memory_utilization)
    available = target_used - current_used - runtime_reserve_bytes
    bytes_per_physical_block = bytes_per_block + workspace_bytes_per_block
    num_blocks = max(0, available // bytes_per_physical_block)
    return KVCacheCapacityPlan(
        total_memory_bytes=total_memory_bytes,
        free_memory_bytes=free_memory_bytes,
        gpu_memory_utilization=gpu_memory_utilization,
        runtime_reserve_bytes=runtime_reserve_bytes,
        bytes_per_block=bytes_per_block,
        workspace_bytes_per_block=workspace_bytes_per_block,
        available_cache_bytes=max(0, available),
        num_blocks=num_blocks,
    )


def plan_kv_cache_capacity(
    model: torch.nn.Module,
    *,
    block_size: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    runtime_reserve_mib: int,
) -> KVCacheCapacityPlan:
    """在模型加载完成后读取真实 CUDA 显存并规划 KV Block。

    ``empty_cache`` 只释放 PyTorch allocator 中没有 Tensor 引用的缓存，不会释放模型
    权重。先清理再读取 ``mem_get_info``，可以避免历史临时分配让“当前占用”虚高。
    """

    parameter = next(model.parameters())
    device = parameter.device
    if device.type != "cuda":
        raise ValueError("自动 KV Block 规划只支持 CUDA 模型")
    if block_size <= 0 or max_num_seqs <= 0:
        raise ValueError("block_size 和 max_num_seqs 必须大于 0")
    if runtime_reserve_mib < 0:
        raise ValueError("runtime_reserve_mib 不能为负数")

    config = model.config
    if not isinstance(config, ModelConfig):
        raise TypeError("model.config 必须是 ModelConfig")
    bytes_per_block = (
        config.num_hidden_layers
        * 2
        * block_size
        * config.num_key_value_heads
        * config.head_dim
        * parameter.element_size()
    )
    # 常驻 GPU BlockTable 的形状是 [max_num_seqs, num_blocks] int32。
    workspace_bytes_per_block = max_num_seqs * 4

    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    plan = calculate_kv_cache_capacity(
        total_memory_bytes=total_bytes,
        free_memory_bytes=free_bytes,
        gpu_memory_utilization=gpu_memory_utilization,
        runtime_reserve_bytes=runtime_reserve_mib * MIB,
        bytes_per_block=bytes_per_block,
        workspace_bytes_per_block=workspace_bytes_per_block,
    )
    if plan.num_blocks <= 0:
        raise RuntimeError(
            "没有足够显存创建一个 KV Block："
            f"{plan.format()}。请提高 gpu_memory_utilization、减小运行余量，"
            "或使用更小模型。"
        )
    return plan

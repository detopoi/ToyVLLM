from __future__ import annotations

"""单 token Decode 的教学版 Triton Paged Attention Kernel。"""

import os
import tempfile

import torch

# Triton 默认写入用户主目录。在受限环境或服务账户下该目录可能不可写，因此把 JIT
# 缓存放到系统临时目录；调用方仍可通过 TRITON_CACHE_DIR 覆盖此默认值。
os.environ.setdefault(
    "TRITON_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "toy-vllm-triton-cache"),
)

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def is_triton_available() -> bool:
    return triton is not None


_AUTOTUNE_NUM_WARPS = (1, 2, 4, 8)
_AUTOTUNE_MIN_IMPROVEMENT = 0.10
_AUTOTUNE_CACHE: dict[tuple[object, ...], int] = {}
_AUTOTUNE_TIMINGS: dict[tuple[object, ...], dict[int, float]] = {}


def clear_triton_autotune_cache() -> None:
    """清除 Python 进程内的配置选择；已编译二进制仍由 Triton 磁盘缓存管理。"""

    _AUTOTUNE_CACHE.clear()
    _AUTOTUNE_TIMINGS.clear()


def get_triton_autotune_results() -> dict[tuple[object, ...], dict[str, object]]:
    """返回调优结果副本，供 benchmark 和教程观察。"""

    return {
        key: {
            "num_warps": _AUTOTUNE_CACHE[key],
            "fastest_num_warps": min(
                _AUTOTUNE_TIMINGS[key],
                key=_AUTOTUNE_TIMINGS[key].get,
            ),
            "timings_ms": dict(_AUTOTUNE_TIMINGS[key]),
        }
        for key in _AUTOTUNE_CACHE
    }


if triton is not None:

    @triton.jit
    def _paged_attention_decode_kernel(
        query_ptr,
        current_key_ptr,
        current_value_ptr,
        key_cache_ptr,
        value_cache_ptr,
        block_tables_ptr,
        context_lengths_ptr,
        output_ptr,
        query_stride_batch: tl.constexpr,
        query_stride_head: tl.constexpr,
        key_stride_batch: tl.constexpr,
        key_stride_head: tl.constexpr,
        cache_stride_layer: tl.constexpr,
        cache_stride_block: tl.constexpr,
        cache_stride_token: tl.constexpr,
        cache_stride_head: tl.constexpr,
        table_stride_batch: tl.constexpr,
        output_stride_batch: tl.constexpr,
        output_stride_head: tl.constexpr,
        layer_index,
        queries_per_kv: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        MAX_BLOCKS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """一个 Program 计算一个 request 的一个 Query Head。

        Program ID 的二维网格是 [batch, query_head]。同一个 Program 在寄存器中保存
        Query、在线 softmax 状态和输出累加器，然后顺序扫描该请求的逻辑 Block。
        """

        batch_index = tl.program_id(0)
        query_head = tl.program_id(1)
        kv_head = query_head // queries_per_kv
        dimensions = tl.arange(0, BLOCK_D)
        dimension_mask = dimensions < HEAD_DIM

        query_offsets = (
            batch_index * query_stride_batch
            + query_head * query_stride_head
            + dimensions
        )
        query = tl.load(
            query_ptr + query_offsets,
            mask=dimension_mask,
            other=0.0,
        ).to(tl.float32)

        current_key_offsets = (
            batch_index * key_stride_batch
            + kv_head * key_stride_head
            + dimensions
        )
        current_key = tl.load(
            current_key_ptr + current_key_offsets,
            mask=dimension_mask,
            other=0.0,
        ).to(tl.float32)
        current_value = tl.load(
            current_value_ptr + current_key_offsets,
            mask=dimension_mask,
            other=0.0,
        ).to(tl.float32)

        # 先把当前 token 放进 softmax，得到一个有限初值。这样后续遇到无效的 padding
        # Block 时，不会出现 max(-inf, -inf) 导致的 NaN。
        scale = 1.0 / tl.sqrt(float(HEAD_DIM))
        running_max = tl.sum(query * current_key, axis=0) * scale
        running_sum = 1.0
        accumulator = current_value

        token_offsets = tl.arange(0, BLOCK_SIZE)
        context_length = tl.load(context_lengths_ptr + batch_index)
        for logical_block in range(MAX_BLOCKS):
            physical_block = tl.load(
                block_tables_ptr
                + batch_index * table_stride_batch
                + logical_block
            )
            logical_token_indices = logical_block * BLOCK_SIZE + token_offsets
            token_mask = logical_token_indices < context_length

            # 物理池布局：
            # [layer, physical_block, block_offset, kv_head, head_dim]
            cache_offsets = (
                layer_index * cache_stride_layer
                + physical_block * cache_stride_block
                + token_offsets[:, None] * cache_stride_token
                + kv_head * cache_stride_head
                + dimensions[None, :]
            )
            matrix_mask = token_mask[:, None] & dimension_mask[None, :]
            block_key = tl.load(
                key_cache_ptr + cache_offsets,
                mask=matrix_mask,
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(block_key * query[None, :], axis=1) * scale
            scores = tl.where(token_mask, scores, float("-inf"))

            block_max = tl.max(scores, axis=0)
            new_max = tl.maximum(running_max, block_max)
            old_scale = tl.exp(running_max - new_max)
            weights = tl.exp(scores - new_max)
            running_sum = running_sum * old_scale + tl.sum(weights, axis=0)

            block_value = tl.load(
                value_cache_ptr + cache_offsets,
                mask=matrix_mask,
                other=0.0,
            ).to(tl.float32)
            accumulator = (
                accumulator * old_scale
                + tl.sum(weights[:, None] * block_value, axis=0)
            )
            running_max = new_max

        output = accumulator / running_sum
        output_offsets = (
            batch_index * output_stride_batch
            + query_head * output_stride_head
            + dimensions
        )
        tl.store(
            output_ptr + output_offsets,
            output,
            mask=dimension_mask,
        )

    @triton.jit
    def _grouped_paged_attention_decode_kernel(
        query_ptr,
        current_key_ptr,
        current_value_ptr,
        key_cache_ptr,
        value_cache_ptr,
        block_tables_ptr,
        context_lengths_ptr,
        output_ptr,
        query_stride_batch: tl.constexpr,
        query_stride_head: tl.constexpr,
        key_stride_batch: tl.constexpr,
        key_stride_head: tl.constexpr,
        cache_stride_layer: tl.constexpr,
        cache_stride_block: tl.constexpr,
        cache_stride_token: tl.constexpr,
        cache_stride_head: tl.constexpr,
        table_stride_batch: tl.constexpr,
        output_stride_batch: tl.constexpr,
        output_stride_head: tl.constexpr,
        layer_index,
        QUERIES_PER_KV: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        MAX_BLOCKS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """一个 Program 同时计算共享同一 KV Head 的一组 Query Head。

        10A 为每个 Query Head 单独读取同一份 K/V。分组后 Grid 的第二维从
        num_query_heads 变成 num_kv_heads，K/V Block 只加载一次，再广播给组内 Query。
        """

        batch_index = tl.program_id(0)
        kv_head = tl.program_id(1)
        group_offsets = tl.arange(0, QUERIES_PER_KV)
        dimensions = tl.arange(0, BLOCK_D)
        dimension_mask = dimensions < HEAD_DIM
        first_query_head = kv_head * QUERIES_PER_KV

        query_offsets = (
            batch_index * query_stride_batch
            + (first_query_head + group_offsets[:, None]) * query_stride_head
            + dimensions[None, :]
        )
        query = tl.load(
            query_ptr + query_offsets,
            mask=dimension_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        current_offsets = (
            batch_index * key_stride_batch
            + kv_head * key_stride_head
            + dimensions
        )
        current_key = tl.load(
            current_key_ptr + current_offsets,
            mask=dimension_mask,
            other=0.0,
        ).to(tl.float32)
        current_value = tl.load(
            current_value_ptr + current_offsets,
            mask=dimension_mask,
            other=0.0,
        ).to(tl.float32)

        scale = 1.0 / tl.sqrt(float(HEAD_DIM))
        running_max = tl.sum(
            query * current_key[None, :],
            axis=1,
        ) * scale
        running_sum = tl.full(
            (QUERIES_PER_KV,),
            1.0,
            tl.float32,
        )
        accumulator = tl.broadcast_to(
            current_value[None, :],
            (QUERIES_PER_KV, BLOCK_D),
        )

        token_offsets = tl.arange(0, BLOCK_SIZE)
        context_length = tl.load(context_lengths_ptr + batch_index)
        for logical_block in range(MAX_BLOCKS):
            physical_block = tl.load(
                block_tables_ptr
                + batch_index * table_stride_batch
                + logical_block
            )
            logical_token_indices = logical_block * BLOCK_SIZE + token_offsets
            token_mask = logical_token_indices < context_length
            cache_offsets = (
                layer_index * cache_stride_layer
                + physical_block * cache_stride_block
                + token_offsets[:, None] * cache_stride_token
                + kv_head * cache_stride_head
                + dimensions[None, :]
            )
            matrix_mask = token_mask[:, None] & dimension_mask[None, :]
            block_key = tl.load(
                key_cache_ptr + cache_offsets,
                mask=matrix_mask,
                other=0.0,
            ).to(tl.float32)

            # [group, token, dim] 沿 dim 归约，得到组内每个 Query 对 Block 各 token 的分数。
            scores = tl.sum(
                query[:, None, :] * block_key[None, :, :],
                axis=2,
            ) * scale
            scores = tl.where(token_mask[None, :], scores, float("-inf"))
            block_max = tl.max(scores, axis=1)
            new_max = tl.maximum(running_max, block_max)
            old_scale = tl.exp(running_max - new_max)
            weights = tl.exp(scores - new_max[:, None])
            running_sum = (
                running_sum * old_scale
                + tl.sum(weights, axis=1)
            )

            block_value = tl.load(
                value_cache_ptr + cache_offsets,
                mask=matrix_mask,
                other=0.0,
            ).to(tl.float32)
            accumulator = (
                accumulator * old_scale[:, None]
                + tl.sum(
                    weights[:, :, None] * block_value[None, :, :],
                    axis=1,
                )
            )
            running_max = new_max

        output = accumulator / running_sum[:, None]
        output_offsets = (
            batch_index * output_stride_batch
            + (first_query_head + group_offsets[:, None]) * output_stride_head
            + dimensions[None, :]
        )
        tl.store(
            output_ptr + output_offsets,
            output,
            mask=dimension_mask[None, :],
        )


def triton_paged_attention(
    query: torch.Tensor,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    *,
    layer_index: int,
    queries_per_kv: int,
    grouped: bool = True,
    autotune: bool = True,
) -> torch.Tensor:
    """启动 Triton Paged Attention。

    输入 Query/K/V 沿用模型布局 `[batch, heads, 1, head_dim]`，输出形状与 Query 相同。
    BlockTable 已经由 Engine 打包为 GPU 上的二维 int32 Tensor。
    """

    if triton is None:
        raise RuntimeError(
            "未安装 Triton；Windows 可安装 triton-windows 后使用此后端"
        )
    if not query.is_cuda:
        raise ValueError("Triton Paged Attention 只支持 CUDA Tensor")
    if query.shape[2] != 1:
        raise ValueError("Triton Paged Attention 只支持单 token Decode")
    if block_tables.ndim != 2 or context_lengths.ndim != 1:
        raise ValueError("BlockTable/Context Length Tensor 形状错误")
    if query.shape[0] != block_tables.shape[0]:
        raise ValueError("Query batch size 与 BlockTable 数量不一致")
    if query.shape[0] != context_lengths.shape[0]:
        raise ValueError("Query batch size 与 Context Length 数量不一致")
    if key_cache.shape != value_cache.shape:
        raise ValueError("Key/Value 物理池形状必须一致")

    batch_size, num_query_heads, _, head_dim = query.shape
    block_size = key_cache.shape[2]
    max_blocks = block_tables.shape[1]
    block_d = triton.next_power_of_2(head_dim)
    if block_d > 256:
        raise ValueError("当前教学 Kernel 仅支持 head_dim <= 256")

    output = torch.empty_like(query)
    num_kv_heads = current_key.shape[1]
    if num_query_heads != num_kv_heads * queries_per_kv:
        raise ValueError("Query Head 与 KV Head 的 GQA 比例不一致")

    kernel = (
        _grouped_paged_attention_decode_kernel
        if grouped
        else _paged_attention_decode_kernel
    )
    grid = (
        batch_size,
        num_kv_heads if grouped else num_query_heads,
    )
    def launch(num_warps: int) -> None:
        kernel[grid](
            query,
            current_key,
            current_value,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
            output,
            query.stride(0),
            query.stride(1),
            current_key.stride(0),
            current_key.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
            block_tables.stride(0),
            output.stride(0),
            output.stride(1),
            layer_index=layer_index,
            **(
                {"QUERIES_PER_KV": queries_per_kv}
                if grouped
                else {"queries_per_kv": queries_per_kv}
            ),
            HEAD_DIM=head_dim,
            BLOCK_SIZE=block_size,
            MAX_BLOCKS=max_blocks,
            BLOCK_D=block_d,
            num_warps=num_warps,
        )

    num_warps = 4
    if autotune and not grouped:
        device_index = query.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        tune_key = (
            device_index,
            query.dtype,
            batch_size,
            num_query_heads,
            head_dim,
            block_size,
            max_blocks,
        )
        if tune_key not in _AUTOTUNE_CACHE:
            timings = {
                candidate: float(
                    triton.testing.do_bench(
                        lambda candidate=candidate: launch(candidate),
                        warmup=20,
                        rep=100,
                        return_mode="median",
                    )
                )
                for candidate in _AUTOTUNE_NUM_WARPS
            }
            _AUTOTUNE_TIMINGS[tune_key] = timings
            fastest = min(timings, key=timings.get)
            baseline = timings[4]
            # 微基准差距太小时容易受缓存、频率和 Windows WDDM 调度噪声影响。
            # 只有至少快 10% 才替换稳定基线，避免“调优后端到端反而更慢”。
            _AUTOTUNE_CACHE[tune_key] = (
                fastest
                if timings[fastest]
                <= baseline * (1.0 - _AUTOTUNE_MIN_IMPROVEMENT)
                else 4
            )
        num_warps = _AUTOTUNE_CACHE[tune_key]

    launch(num_warps)
    return output

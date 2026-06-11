from __future__ import annotations

"""Engine 的 Paged KV Cache 控制面：管理物理块编号和请求 Block Table。

这里不分配任何 GPU Tensor。BlockManager 只管理“哪些物理块空闲、每条请求引用哪些块”，
因此可以在 CPU 上完整测试分配策略。真正的 K/V 数据由 kv_cache.py 中的 PagedKVCache 持有。
"""

from collections import deque
from dataclasses import dataclass

from toyvllm.engine.prefix_cache import PrefixCache, PrefixCacheStats


class OutOfBlocksError(RuntimeError):
    """物理块池容量不足，当前请求不能继续增长。"""


@dataclass(frozen=True)
class PhysicalTokenSlot:
    """一个逻辑 token 最终落到的物理位置。"""

    token_index: int
    physical_block_id: int
    block_offset: int


@dataclass(frozen=True)
class BlockTable:
    """一条请求的逻辑块到物理块映射。

    `physical_block_ids[logical_block_id]` 就是该逻辑块对应的物理块号。
    Block Table 很小，常驻 CPU 即可；真正占显存的是物理 KV Block。
    """

    request_id: int
    block_size: int
    physical_block_ids: tuple[int, ...] = ()
    num_tokens: int = 0

    @property
    def num_blocks(self) -> int:
        return len(self.physical_block_ids)

    @property
    def capacity(self) -> int:
        return self.num_blocks * self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        if self.num_tokens == 0:
            return 0
        remainder = self.num_tokens % self.block_size
        return self.block_size if remainder == 0 else remainder

    def slots(self, start: int = 0, end: int | None = None) -> tuple[PhysicalTokenSlot, ...]:
        """把逻辑 token 区间翻译成物理块号和块内偏移。"""

        stop = self.num_tokens if end is None else end
        if not 0 <= start <= stop <= self.num_tokens:
            raise ValueError("token 区间必须位于 [0, num_tokens] 内")

        result = []
        for token_index in range(start, stop):
            logical_block_id, block_offset = divmod(
                token_index,
                self.block_size,
            )
            result.append(
                PhysicalTokenSlot(
                    token_index=token_index,
                    physical_block_id=self.physical_block_ids[logical_block_id],
                    block_offset=block_offset,
                )
            )
        return tuple(result)


@dataclass(frozen=True)
class BlockManagerStats:
    num_total_blocks: int
    num_free_blocks: int
    num_allocatable_blocks: int
    num_used_blocks: int
    num_requests: int
    num_cached_blocks: int = 0
    num_evictable_cached_blocks: int = 0
    num_shared_blocks: int = 0
    num_prefix_cache_hits: int = 0
    num_prefix_cache_hit_tokens: int = 0
    num_prefix_cache_evictions: int = 0


class BlockManager:
    """固定大小物理块的分配器。

    重要不变量：

    - 普通块只属于一条请求，完整前缀块可以由多条请求只读共享；
    - 请求只按实际增长分配块，不预留最大上下文；
    - reserve 要么完整成功，要么完全不修改 Block Table；
    - free 只减少引用；缓存块会继续驻留，内存紧张时再按 LRU 淘汰；
    - 只共享完整块，因此后续写入总从新块开始，不需要 Copy-on-Write。
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        *,
        enable_prefix_cache: bool = False,
    ) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks 必须大于 0")
        if block_size <= 0:
            raise ValueError("block_size 必须大于 0")

        self.num_blocks = num_blocks
        self.block_size = block_size

        # 物理块只是整数编号。真实 Tensor 会按相同编号预分配为一个大块池。
        self._free_blocks: deque[int] = deque(range(num_blocks))
        self._tables: dict[int, BlockTable] = {}
        # ref_count 只统计活跃 Block Table 的引用。Prefix Cache 本身相当于一个“软引用”：
        # ref_count=0 时 KV 仍驻留，但可以在分配压力下被 LRU 淘汰。
        self._block_ref_counts = [0] * num_blocks
        self._prefix_cache = (
            PrefixCache(block_size) if enable_prefix_cache else None
        )

    def allocate(
        self,
        request_id: int,
        num_tokens: int = 0,
        *,
        capacity_tokens: int | None = None,
    ) -> BlockTable:
        """注册请求，并可让物理容量大于当前有效 token 数。

        Chunked Prefill 接纳请求时会为完整 Prompt 预留容量，但 ``num_tokens`` 保持 0。
        后续每完成一个 chunk，再通过 reserve 推进有效长度。这样既不会读取未写入槽位，
        也避免多个半完成 Prompt 占满运行槽后因物理块不足互相等待。
        """

        if request_id in self._tables:
            raise ValueError(f"request_id 已存在：{request_id}")
        if num_tokens < 0:
            raise ValueError("num_tokens 不能为负数")
        if capacity_tokens is None:
            capacity_tokens = num_tokens
        if capacity_tokens < num_tokens:
            raise ValueError("capacity_tokens 不能小于 num_tokens")

        required_blocks = self._blocks_for_tokens(capacity_tokens)
        self._require_free_blocks(required_blocks)

        # 容量检查完成后才修改内部状态，保证失败时没有半初始化请求。
        allocated = self._take_blocks(required_blocks)
        table = BlockTable(
            request_id=request_id,
            block_size=self.block_size,
            physical_block_ids=allocated,
            num_tokens=num_tokens,
        )
        self._tables[request_id] = table
        return table

    def reserve(
        self,
        request_id: int,
        num_new_tokens: int,
    ) -> tuple[PhysicalTokenSlot, ...]:
        """为请求追加 token 空间，只在跨过块边界时分配新物理块。

        返回值只描述本次新增 token 的物理位置，Executor 可以直接把本轮产生的 K/V
        写入这些槽位，不必重新扫描整条请求。
        """

        if num_new_tokens <= 0:
            raise ValueError("num_new_tokens 必须大于 0")
        old_table = self.get_block_table(request_id)
        new_num_tokens = old_table.num_tokens + num_new_tokens
        required_total = self._blocks_for_tokens(new_num_tokens)
        additional_blocks = max(0, required_total - old_table.num_blocks)

        # 先检查、后分配、最后替换不可变 BlockTable。任何异常都不会改变旧映射。
        self._require_free_blocks(additional_blocks)
        allocated = self._take_blocks(additional_blocks)
        new_table = BlockTable(
            request_id=request_id,
            block_size=self.block_size,
            physical_block_ids=old_table.physical_block_ids + allocated,
            num_tokens=new_num_tokens,
        )
        self._tables[request_id] = new_table
        return new_table.slots(old_table.num_tokens, new_num_tokens)

    def reserve_many(
        self,
        requests: tuple[tuple[int, int], ...],
    ) -> dict[int, tuple[PhysicalTokenSlot, ...]]:
        """原子地为同一 Decode Batch 中的多条请求预留空间。

        先汇总所有请求本轮需要的新物理块，确认总量足够后才逐条 reserve。
        如果只对每条请求分别调用 can_reserve，多条请求可能都看到同一批空闲块并误判成功。
        """

        if not requests:
            return {}
        request_ids = [request_id for request_id, _ in requests]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("reserve_many 中 request_id 不能重复")

        total_additional_blocks = 0
        for request_id, num_new_tokens in requests:
            if num_new_tokens <= 0:
                raise ValueError("num_new_tokens 必须大于 0")
            table = self.get_block_table(request_id)
            required_total = self._blocks_for_tokens(
                table.num_tokens + num_new_tokens
            )
            total_additional_blocks += max(
                0,
                required_total - table.num_blocks,
            )
        self._require_free_blocks(total_additional_blocks)

        return {
            request_id: self.reserve(request_id, num_new_tokens)
            for request_id, num_new_tokens in requests
        }

    def free(self, request_id: int) -> tuple[int, ...]:
        """释放请求的全部物理块，并删除 Block Table。"""

        try:
            table = self._tables.pop(request_id)
        except KeyError as error:
            raise KeyError(f"未知 request_id：{request_id}") from error

        # 共享块不能直接放回空闲队列。最后一条活跃请求结束后，普通块立即回收；
        # Prefix Cache 块继续保留 K/V，直到缓存被清空或内存压力触发 LRU 淘汰。
        for block_id in table.physical_block_ids:
            self._block_ref_counts[block_id] -= 1
            if self._block_ref_counts[block_id] < 0:
                raise RuntimeError(f"物理 Block {block_id} 引用计数变成负数")
            if (
                self._block_ref_counts[block_id] == 0
                and not self._is_cached_block(block_id)
            ):
                self._free_blocks.append(block_id)
        return table.physical_block_ids

    def can_reserve(self, request_id: int, num_new_tokens: int) -> bool:
        """在不修改状态的情况下判断请求能否增长。"""

        if num_new_tokens <= 0:
            return False
        table = self.get_block_table(request_id)
        required_total = self._blocks_for_tokens(
            table.num_tokens + num_new_tokens
        )
        additional = max(0, required_total - table.num_blocks)
        return additional <= self._num_allocatable_blocks()

    def additional_blocks_required(
        self,
        request_id: int,
        num_new_tokens: int,
    ) -> int:
        """返回增长所需的新 Block 数，不修改 Block Table。"""

        if num_new_tokens <= 0:
            raise ValueError("num_new_tokens 必须大于 0")
        table = self.get_block_table(request_id)
        required_total = self._blocks_for_tokens(
            table.num_tokens + num_new_tokens
        )
        return max(0, required_total - table.num_blocks)

    def max_reservable_tokens(self, request_id: int) -> int:
        """当前空闲块全部给该请求时，它最多还能增长多少 token。"""

        table = self.get_block_table(request_id)
        total_capacity = (
            table.num_blocks + self._num_allocatable_blocks()
        ) * self.block_size
        return total_capacity - table.num_tokens

    def attach_cached_prefix(
        self,
        request_id: int,
        token_ids: list[int],
    ) -> int:
        """把最长连续缓存前缀挂到空 Block Table，返回复用的 token 数。

        最后一个 Prompt token 必须实际执行一次前向，才能得到首个输出 token 的 logits。
        因此即使整个 Prompt 都由完整块组成，也最多复用“最后一个 token 之前”的完整块。
        """

        if self._prefix_cache is None:
            return 0
        table = self.get_block_table(request_id)
        if table.num_tokens or table.physical_block_ids:
            raise RuntimeError("只有空 Block Table 可以挂载 Prefix Cache")

        max_cacheable_tokens = max(0, len(token_ids) - 1)
        block_ids: list[int] = []
        matched_tokens = 0
        for key, _, block_tokens, end in self._prefix_cache.iter_full_blocks(
            token_ids,
            max_tokens=max_cacheable_tokens,
        ):
            entry = self._prefix_cache.get(
                key,
                block_tokens,
                record_hit=True,
            )
            if entry is None:
                break
            block_ids.append(entry.block_id)
            self._block_ref_counts[entry.block_id] += 1
            matched_tokens = end

        if block_ids:
            self._tables[request_id] = BlockTable(
                request_id=request_id,
                block_size=self.block_size,
                physical_block_ids=tuple(block_ids),
                num_tokens=matched_tokens,
            )
        return matched_tokens

    def cached_prefix_tokens(self, token_ids: list[int]) -> int:
        """只预览最长可复用前缀，不修改引用计数和命中统计。"""

        if self._prefix_cache is None:
            return 0
        matched_tokens = 0
        for key, _, block_tokens, end in self._prefix_cache.iter_full_blocks(
            token_ids,
            max_tokens=max(0, len(token_ids) - 1),
        ):
            if self._prefix_cache.get(
                key,
                block_tokens,
                record_hit=False,
            ) is None:
                break
            matched_tokens = end
        return matched_tokens

    def cache_computed_prompt_blocks(
        self,
        request_id: int,
        prompt_token_ids: list[int],
    ) -> int:
        """登记已经完成计算的 Prompt 整块，返回本次新增缓存块数。

        Engine 必须在 K/V 写回 GPU 之后调用本方法。BlockManager 只验证 Block Table
        的有效长度，无法自行判断真实 Tensor 是否已经完成写入。
        """

        if self._prefix_cache is None:
            return 0
        table = self.get_block_table(request_id)
        max_tokens = min(table.num_tokens, len(prompt_token_ids))
        inserted = 0
        for logical_block_id, (
            key,
            parent_key,
            block_tokens,
            _,
        ) in enumerate(
            self._prefix_cache.iter_full_blocks(
                prompt_token_ids,
                max_tokens=max_tokens,
            )
        ):
            existing = self._prefix_cache.get(
                key,
                block_tokens,
                record_hit=False,
            )
            if existing is not None:
                continue
            block_id = table.physical_block_ids[logical_block_id]
            if self._prefix_cache.insert(
                key=key,
                parent_key=parent_key,
                token_ids=block_tokens,
                block_id=block_id,
            ):
                inserted += 1
        return inserted

    def clear_prefix_cache(self) -> None:
        """移除全部缓存索引；没有活跃引用的物理块立即回到空闲队列。"""

        if self._prefix_cache is None:
            return
        for entry in self._prefix_cache.clear():
            if self._block_ref_counts[entry.block_id] == 0:
                self._free_blocks.append(entry.block_id)

    def blocks_required_for_tokens(self, num_tokens: int) -> int:
        """公开只读容量计算，供 Scheduler admission 使用。"""

        if num_tokens < 0:
            raise ValueError("num_tokens 不能为负数")
        return self._blocks_for_tokens(num_tokens)

    def get_block_table(self, request_id: int) -> BlockTable:
        try:
            return self._tables[request_id]
        except KeyError as error:
            raise KeyError(f"未知 request_id：{request_id}") from error

    @property
    def free_block_ids(self) -> tuple[int, ...]:
        return tuple(self._free_blocks)

    @property
    def stats(self) -> BlockManagerStats:
        free = len(self._free_blocks)
        cache_stats = self.prefix_cache_stats
        return BlockManagerStats(
            num_total_blocks=self.num_blocks,
            num_free_blocks=free,
            num_allocatable_blocks=self._num_allocatable_blocks(),
            num_used_blocks=self.num_blocks - free,
            num_requests=len(self._tables),
            num_cached_blocks=cache_stats.num_cached_blocks,
            num_evictable_cached_blocks=cache_stats.num_evictable_blocks,
            num_shared_blocks=sum(
                count > 1 for count in self._block_ref_counts
            ),
            num_prefix_cache_hits=cache_stats.num_hits,
            num_prefix_cache_hit_tokens=cache_stats.num_hit_tokens,
            num_prefix_cache_evictions=cache_stats.num_evictions,
        )

    @property
    def prefix_cache_stats(self) -> PrefixCacheStats:
        if self._prefix_cache is None:
            return PrefixCacheStats(0, 0, 0, 0, 0)
        return self._prefix_cache.stats(
            lambda block_id: self._block_ref_counts[block_id] == 0
        )

    def _blocks_for_tokens(self, num_tokens: int) -> int:
        # ceil(num_tokens / block_size)，但整数公式不会引入浮点数。
        return (num_tokens + self.block_size - 1) // self.block_size

    def _require_free_blocks(self, required_blocks: int) -> None:
        available = self._num_allocatable_blocks()
        if required_blocks > available:
            raise OutOfBlocksError(
                f"KV Block 不足：需要 {required_blocks}，可分配 {available}"
            )
        while len(self._free_blocks) < required_blocks:
            if self._prefix_cache is None:
                raise RuntimeError("可分配块统计与空闲队列不一致")
            entry = self._prefix_cache.evict_one(
                lambda block_id: self._block_ref_counts[block_id] == 0
            )
            if entry is None:
                raise RuntimeError("Prefix Cache 没有可淘汰块")
            self._free_blocks.append(entry.block_id)

    def _take_blocks(self, count: int) -> tuple[int, ...]:
        self._require_free_blocks(count)
        block_ids = tuple(self._free_blocks.popleft() for _ in range(count))
        for block_id in block_ids:
            if self._block_ref_counts[block_id] != 0:
                raise RuntimeError(f"分配到仍有引用的物理 Block {block_id}")
            self._block_ref_counts[block_id] = 1
        return block_ids

    def _is_cached_block(self, block_id: int) -> bool:
        return (
            self._prefix_cache is not None
            and self._prefix_cache.contains_block(block_id)
        )

    def _num_allocatable_blocks(self) -> int:
        available = len(self._free_blocks)
        if self._prefix_cache is not None:
            available += self._prefix_cache.count_evictable(
                lambda block_id: self._block_ref_counts[block_id] == 0
            )
        return available

from __future__ import annotations

"""Paged KV Cache 的控制面：管理物理块编号和请求 Block Table。

这里不分配任何 GPU Tensor。BlockManager 只管理“哪些物理块空闲、每条请求引用哪些块”，
因此可以在 CPU 上完整测试分配策略。真正的 K/V 数据由 kv_cache.py 中的 PagedKVCache 持有。
"""

from collections import deque
from dataclasses import dataclass


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
    num_used_blocks: int
    num_requests: int


class BlockManager:
    """固定大小物理块的分配器。

    重要不变量：

    - 一个物理块同一时间只属于一条请求；
    - 请求只按实际增长分配块，不预留最大上下文；
    - reserve 要么完整成功，要么完全不修改 Block Table；
    - free 后物理块回到空闲队列，可以被后续请求复用。

    当前版本还不实现共享前缀和 Copy-on-Write，所以没有引用计数。共享只表示“请求结束后
    物理块可被别的请求复用”，不是两条活跃请求同时引用同一块。
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks 必须大于 0")
        if block_size <= 0:
            raise ValueError("block_size 必须大于 0")

        self.num_blocks = num_blocks
        self.block_size = block_size

        # 物理块只是整数编号。真实 Tensor 会按相同编号预分配为一个大块池。
        self._free_blocks: deque[int] = deque(range(num_blocks))
        self._tables: dict[int, BlockTable] = {}

    def allocate(self, request_id: int, num_tokens: int = 0) -> BlockTable:
        """注册新请求，并可一次性为 Prompt 预留所需块。"""

        if request_id in self._tables:
            raise ValueError(f"request_id 已存在：{request_id}")
        if num_tokens < 0:
            raise ValueError("num_tokens 不能为负数")

        required_blocks = self._blocks_for_tokens(num_tokens)
        self._require_free_blocks(required_blocks)

        # 容量检查完成后才修改内部状态，保证失败时没有半初始化请求。
        allocated = tuple(self._free_blocks.popleft() for _ in range(required_blocks))
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
        additional_blocks = required_total - old_table.num_blocks

        # 先检查、后分配、最后替换不可变 BlockTable。任何异常都不会改变旧映射。
        self._require_free_blocks(additional_blocks)
        allocated = tuple(
            self._free_blocks.popleft() for _ in range(additional_blocks)
        )
        new_table = BlockTable(
            request_id=request_id,
            block_size=self.block_size,
            physical_block_ids=old_table.physical_block_ids + allocated,
            num_tokens=new_num_tokens,
        )
        self._tables[request_id] = new_table
        return new_table.slots(old_table.num_tokens, new_num_tokens)

    def free(self, request_id: int) -> tuple[int, ...]:
        """释放请求的全部物理块，并删除 Block Table。"""

        try:
            table = self._tables.pop(request_id)
        except KeyError as error:
            raise KeyError(f"未知 request_id：{request_id}") from error

        # 使用 FIFO 空闲队列。刚释放块放到队尾，所有块会得到较均匀的复用机会。
        self._free_blocks.extend(table.physical_block_ids)
        return table.physical_block_ids

    def can_reserve(self, request_id: int, num_new_tokens: int) -> bool:
        """在不修改状态的情况下判断请求能否增长。"""

        if num_new_tokens <= 0:
            return False
        table = self.get_block_table(request_id)
        required_total = self._blocks_for_tokens(
            table.num_tokens + num_new_tokens
        )
        additional = required_total - table.num_blocks
        return additional <= len(self._free_blocks)

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
        return BlockManagerStats(
            num_total_blocks=self.num_blocks,
            num_free_blocks=free,
            num_used_blocks=self.num_blocks - free,
            num_requests=len(self._tables),
        )

    def _blocks_for_tokens(self, num_tokens: int) -> int:
        # ceil(num_tokens / block_size)，但整数公式不会引入浮点数。
        return (num_tokens + self.block_size - 1) // self.block_size

    def _require_free_blocks(self, required_blocks: int) -> None:
        available = len(self._free_blocks)
        if required_blocks > available:
            raise OutOfBlocksError(
                f"KV Block 不足：需要 {required_blocks}，空闲 {available}"
            )


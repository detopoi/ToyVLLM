from __future__ import annotations

"""Prefix Cache 的 CPU 元数据。

真实 K/V 仍保存在 GPU 的 PagedKVCache 中。这里仅记录：

    token 前缀 -> 物理 KV Block 编号

缓存粒度必须是完整 Block，因为 Block Table 只能共享完整物理块。若共享半块，
新请求继续写入该块时就会覆盖旧请求的 K/V，需要额外实现 Copy-on-Write。
"""

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import struct
from typing import Callable, Iterator, Sequence


@dataclass(frozen=True)
class PrefixCacheEntry:
    """一个完整 token block 对应的物理 KV Block。"""

    key: bytes
    parent_key: bytes
    token_ids: tuple[int, ...]
    block_id: int


@dataclass(frozen=True)
class PrefixCacheStats:
    num_cached_blocks: int
    num_evictable_blocks: int
    num_hits: int
    num_hit_tokens: int
    num_evictions: int


class PrefixCache:
    """使用链式摘要标识完整前缀，并以 LRU 顺序保存缓存项。

    第 N 块的 key 同时包含“前 N-1 块的摘要”和“第 N 块 token”，因此相同 token
    只有在前面的上下文也相同时才会命中。单独用当前 block 的 token 做 key 会把
    出现在不同位置、不同上下文中的相同文本错误地当成同一份 KV。
    """

    ROOT_KEY = bytes(32)

    def __init__(self, block_size: int) -> None:
        self.block_size = block_size
        self._entries: dict[bytes, PrefixCacheEntry] = {}
        self._block_keys: dict[int, bytes] = {}
        self._lru: OrderedDict[bytes, None] = OrderedDict()
        self.num_hits = 0
        self.num_hit_tokens = 0
        self.num_evictions = 0

    def iter_full_blocks(
        self,
        token_ids: Sequence[int],
        *,
        max_tokens: int | None = None,
    ) -> Iterator[tuple[bytes, bytes, tuple[int, ...], int]]:
        """依次产生完整 block 的链式 key，返回值中的 end 是累计 token 数。"""

        stop = len(token_ids) if max_tokens is None else min(len(token_ids), max_tokens)
        stop -= stop % self.block_size
        parent_key = self.ROOT_KEY
        for start in range(0, stop, self.block_size):
            block_tokens = tuple(token_ids[start : start + self.block_size])
            key = self._make_key(parent_key, block_tokens)
            yield key, parent_key, block_tokens, start + self.block_size
            parent_key = key

    def get(
        self,
        key: bytes,
        token_ids: tuple[int, ...],
        *,
        record_hit: bool,
    ) -> PrefixCacheEntry | None:
        entry = self._entries.get(key)
        # SHA-256 碰撞几乎不可见，但比较原 token 能让正确性不依赖概率假设。
        if entry is None or entry.token_ids != token_ids:
            return None
        self._touch(key)
        if record_hit:
            self.num_hits += 1
            self.num_hit_tokens += self.block_size
        return entry

    def insert(
        self,
        *,
        key: bytes,
        parent_key: bytes,
        token_ids: tuple[int, ...],
        block_id: int,
    ) -> bool:
        """登记已写完的物理块；相同前缀由最早登记的块作为规范副本。"""

        existing = self._entries.get(key)
        if existing is not None:
            self._touch(key)
            return False
        if block_id in self._block_keys:
            raise RuntimeError(f"物理 Block {block_id} 已属于另一个 Prefix Cache 项")
        entry = PrefixCacheEntry(
            key=key,
            parent_key=parent_key,
            token_ids=token_ids,
            block_id=block_id,
        )
        self._entries[key] = entry
        self._block_keys[block_id] = key
        self._lru[key] = None
        return True

    def contains_block(self, block_id: int) -> bool:
        return block_id in self._block_keys

    def count_evictable(self, can_evict: Callable[[int], bool]) -> int:
        return sum(
            1 for entry in self._entries.values() if can_evict(entry.block_id)
        )

    def evict_one(
        self,
        can_evict: Callable[[int], bool],
    ) -> PrefixCacheEntry | None:
        """淘汰最久未使用、且没有活跃请求引用的缓存块。"""

        for key in tuple(self._lru):
            entry = self._entries[key]
            if not can_evict(entry.block_id):
                continue
            self.remove(key)
            self.num_evictions += 1
            return entry
        return None

    def remove(self, key: bytes) -> PrefixCacheEntry:
        entry = self._entries.pop(key)
        del self._block_keys[entry.block_id]
        del self._lru[key]
        return entry

    def clear(self) -> tuple[PrefixCacheEntry, ...]:
        entries = tuple(self._entries.values())
        self._entries.clear()
        self._block_keys.clear()
        self._lru.clear()
        return entries

    def stats(self, can_evict: Callable[[int], bool]) -> PrefixCacheStats:
        return PrefixCacheStats(
            num_cached_blocks=len(self._entries),
            num_evictable_blocks=self.count_evictable(can_evict),
            num_hits=self.num_hits,
            num_hit_tokens=self.num_hit_tokens,
            num_evictions=self.num_evictions,
        )

    def _touch(self, key: bytes) -> None:
        self._lru.move_to_end(key)

    @staticmethod
    def _make_key(parent_key: bytes, token_ids: tuple[int, ...]) -> bytes:
        payload = struct.pack(f"<{len(token_ids)}q", *token_ids)
        return hashlib.sha256(parent_key + payload).digest()

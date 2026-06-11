import unittest

from toyvllm.engine import BlockManager, OutOfBlocksError


class BlockManagerTest(unittest.TestCase):
    def test_capacity_can_be_reserved_before_tokens_become_valid(self) -> None:
        manager = BlockManager(num_blocks=4, block_size=2)
        table = manager.allocate(1, num_tokens=0, capacity_tokens=5)

        self.assertEqual(table.num_tokens, 0)
        self.assertEqual(table.num_blocks, 3)
        slots = manager.reserve(1, 2)
        self.assertEqual(len(slots), 2)
        self.assertEqual(manager.get_block_table(1).num_tokens, 2)
        self.assertEqual(manager.stats.num_used_blocks, 3)

    def test_allocate_and_append_across_block_boundary(self) -> None:
        manager = BlockManager(num_blocks=4, block_size=4)
        table = manager.allocate(request_id=10, num_tokens=3)
        self.assertEqual(table.physical_block_ids, (0,))
        self.assertEqual(table.last_block_num_tokens, 3)

        slots = manager.reserve(10, 3)
        table = manager.get_block_table(10)
        self.assertEqual(table.physical_block_ids, (0, 1))
        self.assertEqual(
            [(slot.physical_block_id, slot.block_offset) for slot in slots],
            [(0, 3), (1, 0), (1, 1)],
        )

    def test_free_blocks_are_reused_by_later_request(self) -> None:
        manager = BlockManager(num_blocks=2, block_size=4)
        first = manager.allocate(request_id=1, num_tokens=8)
        self.assertEqual(first.physical_block_ids, (0, 1))
        manager.free(1)

        second = manager.allocate(request_id=2, num_tokens=5)
        self.assertEqual(second.physical_block_ids, (0, 1))
        self.assertEqual(manager.stats.num_free_blocks, 0)

    def test_out_of_blocks_is_atomic(self) -> None:
        manager = BlockManager(num_blocks=2, block_size=4)
        original = manager.allocate(request_id=1, num_tokens=4)

        with self.assertRaises(OutOfBlocksError):
            manager.reserve(1, 5)

        # 失败后旧 Block Table 和空闲队列必须完全不变。
        self.assertEqual(manager.get_block_table(1), original)
        self.assertEqual(manager.free_block_ids, (1,))

    def test_non_contiguous_physical_blocks_form_one_logical_sequence(self) -> None:
        manager = BlockManager(num_blocks=4, block_size=2)
        manager.allocate(request_id=100, num_tokens=2)  # 占物理块 0
        request = manager.allocate(request_id=200, num_tokens=2)  # 占物理块 1
        manager.allocate(request_id=300, num_tokens=2)  # 占物理块 2
        manager.free(100)  # 空闲队列现在是 [3, 0]

        manager.reserve(200, 2)
        request = manager.get_block_table(200)
        self.assertEqual(request.physical_block_ids, (1, 3))
        self.assertEqual(
            [slot.physical_block_id for slot in request.slots()],
            [1, 1, 3, 3],
        )

    def test_can_reserve_does_not_modify_state(self) -> None:
        manager = BlockManager(num_blocks=2, block_size=4)
        original = manager.allocate(request_id=1, num_tokens=3)
        self.assertTrue(manager.can_reserve(1, 1))
        self.assertTrue(manager.can_reserve(1, 5))
        self.assertFalse(manager.can_reserve(1, 6))
        self.assertEqual(manager.get_block_table(1), original)

    def test_reserve_many_is_atomic_across_requests(self) -> None:
        manager = BlockManager(num_blocks=3, block_size=2)
        first = manager.allocate(1, num_tokens=2)
        second = manager.allocate(2, num_tokens=2)

        with self.assertRaises(OutOfBlocksError):
            manager.reserve_many(((1, 3), (2, 1)))

        self.assertEqual(manager.get_block_table(1), first)
        self.assertEqual(manager.get_block_table(2), second)
        self.assertEqual(manager.free_block_ids, (2,))

    def test_prefix_cache_reuses_complete_blocks_and_keeps_tail_for_logits(self) -> None:
        manager = BlockManager(
            num_blocks=4,
            block_size=2,
            enable_prefix_cache=True,
        )
        prompt = [10, 11, 12, 13, 14]

        manager.allocate(1)
        manager.reserve(1, len(prompt))
        self.assertEqual(manager.cache_computed_prompt_blocks(1, prompt), 2)
        first_blocks = manager.get_block_table(1).physical_block_ids
        manager.free(1)

        manager.allocate(2)
        hit_tokens = manager.attach_cached_prefix(2, prompt)
        second = manager.get_block_table(2)

        # 长度 5 的 Prompt 可复用前 4 个 token；最后 token 必须重算以产生 logits。
        self.assertEqual(hit_tokens, 4)
        self.assertEqual(second.physical_block_ids, first_blocks[:2])
        self.assertEqual(second.num_tokens, 4)
        self.assertEqual(manager.stats.num_prefix_cache_hit_tokens, 4)

    def test_prefix_cache_does_not_reuse_the_final_prompt_block(self) -> None:
        manager = BlockManager(
            num_blocks=3,
            block_size=2,
            enable_prefix_cache=True,
        )
        prompt = [1, 2, 3, 4]
        manager.allocate(1)
        manager.reserve(1, len(prompt))
        manager.cache_computed_prompt_blocks(1, prompt)
        manager.free(1)

        manager.allocate(2)
        self.assertEqual(manager.attach_cached_prefix(2, prompt), 2)

    def test_prefix_key_depends_on_all_previous_blocks(self) -> None:
        manager = BlockManager(
            num_blocks=4,
            block_size=2,
            enable_prefix_cache=True,
        )
        manager.allocate(1)
        manager.reserve(1, 4)
        manager.cache_computed_prompt_blocks(1, [1, 2, 3, 4])
        manager.free(1)

        manager.allocate(2)
        # 第二个 block 虽然同为 [3, 4]，但前置上下文不同，不能从中间开始命中。
        self.assertEqual(
            manager.attach_cached_prefix(2, [8, 9, 3, 4, 5]),
            0,
        )

    def test_lru_cache_block_is_evicted_when_allocator_needs_space(self) -> None:
        manager = BlockManager(
            num_blocks=2,
            block_size=2,
            enable_prefix_cache=True,
        )
        manager.allocate(1)
        manager.reserve(1, 2)
        manager.cache_computed_prompt_blocks(1, [1, 2])
        manager.free(1)
        self.assertEqual(manager.stats.num_free_blocks, 1)
        self.assertEqual(manager.stats.num_evictable_cached_blocks, 1)

        # 两块新容量会消耗普通空闲块，并淘汰已经没有活跃引用的缓存块。
        manager.allocate(2, num_tokens=4)
        self.assertEqual(manager.stats.num_prefix_cache_evictions, 1)
        self.assertEqual(manager.stats.num_cached_blocks, 0)

    def test_active_shared_cache_block_cannot_be_evicted(self) -> None:
        manager = BlockManager(
            num_blocks=2,
            block_size=2,
            enable_prefix_cache=True,
        )
        prompt = [1, 2, 3]
        manager.allocate(1)
        manager.reserve(1, 2)
        manager.cache_computed_prompt_blocks(1, prompt)
        manager.free(1)

        manager.allocate(2)
        self.assertEqual(manager.attach_cached_prefix(2, prompt), 2)
        manager.allocate(3, num_tokens=2)

        with self.assertRaises(OutOfBlocksError):
            manager.allocate(4, num_tokens=2)


if __name__ == "__main__":
    unittest.main()

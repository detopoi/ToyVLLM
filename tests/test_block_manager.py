import unittest

from toyvllm.block_manager import BlockManager, OutOfBlocksError


class BlockManagerTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

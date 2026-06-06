import unittest

import torch

from toyvllm.block_manager import BlockManager
from toyvllm.kv_cache import PagedKVCache


def make_layer_cache(
    *,
    num_layers: int,
    num_heads: int,
    num_tokens: int,
    head_dim: int,
    start: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    layers = []
    elements = num_heads * num_tokens * head_dim
    for layer in range(num_layers):
        values = torch.arange(
            start + layer * elements,
            start + (layer + 1) * elements,
            dtype=torch.float32,
        ).view(1, num_heads, num_tokens, head_dim)
        layers.append((values, values + 10_000))
    return layers


class PagedKVCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = BlockManager(num_blocks=4, block_size=2)
        self.cache = PagedKVCache(
            num_layers=2,
            num_blocks=4,
            block_size=2,
            num_kv_heads=2,
            head_dim=3,
            dtype=torch.float32,
            device="cpu",
        )

    def test_write_and_read_round_trip_across_non_contiguous_blocks(self) -> None:
        self.manager.allocate(100, num_tokens=2)  # block 0
        self.manager.allocate(200, num_tokens=2)  # block 1
        self.manager.allocate(300, num_tokens=2)  # block 2
        self.manager.free(100)  # free queue: [3, 0]
        slots = self.manager.reserve(200, 2)  # request 200 gets block 3

        first = make_layer_cache(
            num_layers=2,
            num_heads=2,
            num_tokens=2,
            head_dim=3,
            start=0,
        )
        second = make_layer_cache(
            num_layers=2,
            num_heads=2,
            num_tokens=2,
            head_dim=3,
            start=100,
        )
        table = self.manager.get_block_table(200)
        self.cache.write(table.slots(0, 2), first)
        self.cache.write(slots, second)

        restored = self.cache.read(table)
        for layer in range(2):
            expected_key = torch.cat((first[layer][0], second[layer][0]), dim=2)
            expected_value = torch.cat((first[layer][1], second[layer][1]), dim=2)
            torch.testing.assert_close(restored[layer][0], expected_key)
            torch.testing.assert_close(restored[layer][1], expected_value)

    def test_clear_blocks_removes_old_data(self) -> None:
        table = self.manager.allocate(1, num_tokens=2)
        values = make_layer_cache(
            num_layers=2,
            num_heads=2,
            num_tokens=2,
            head_dim=3,
            start=1,
        )
        self.cache.write(table.slots(), values)
        self.cache.clear_blocks(table.physical_block_ids)
        self.assertEqual(
            torch.count_nonzero(
                self.cache.key_cache[:, list(table.physical_block_ids)]
            ).item(),
            0,
        )

    def test_block_memory_size(self) -> None:
        # 2 layers * 2 tokens * 2 heads * 3 dim * 4 bytes * K/V
        self.assertEqual(self.cache.bytes_per_block, 192)
        self.assertEqual(self.cache.allocated_bytes, 768)

    def test_invalid_write_shape_is_rejected(self) -> None:
        table = self.manager.allocate(1, num_tokens=1)
        invalid = [(torch.zeros(1, 2, 2, 3), torch.zeros(1, 2, 2, 3))] * 2
        with self.assertRaises(ValueError):
            self.cache.write(table.slots(), invalid)

    def test_batch_read_and_incremental_decode_write(self) -> None:
        first_table = self.manager.allocate(1, num_tokens=2)
        second_table = self.manager.allocate(2, num_tokens=1)
        first = make_layer_cache(
            num_layers=2,
            num_heads=2,
            num_tokens=2,
            head_dim=3,
            start=0,
        )
        second = make_layer_cache(
            num_layers=2,
            num_heads=2,
            num_tokens=1,
            head_dim=3,
            start=100,
        )
        self.cache.write(first_table.slots(), first)
        self.cache.write(second_table.slots(), second)

        packed, mask = self.cache.read_batch((first_table, second_table))
        self.assertEqual(mask.tolist(), [[1, 1], [0, 1]])
        self.assertEqual(packed[0][0].shape, (2, 2, 2, 3))

        slots = self.manager.reserve_many(((1, 1), (2, 1)))
        present = []
        for layer in range(2):
            key = torch.cat(
                (
                    packed[layer][0],
                    torch.full((2, 2, 1, 3), 500 + layer),
                ),
                dim=2,
            )
            value = torch.cat(
                (
                    packed[layer][1],
                    torch.full((2, 2, 1, 3), 900 + layer),
                ),
                dim=2,
            )
            present.append((key, value))
        self.cache.write_decode_batch(
            (slots[1], slots[2]),
            present,
        )

        restored_first = self.cache.read(self.manager.get_block_table(1))
        restored_second = self.cache.read(self.manager.get_block_table(2))
        self.assertTrue(torch.all(restored_first[0][0][:, :, -1] == 500))
        self.assertTrue(torch.all(restored_second[1][1][:, :, -1] == 901))

    def test_vectorized_decode_write_matches_reference_path(self) -> None:
        first_table = self.manager.allocate(1, num_tokens=1)
        second_table = self.manager.allocate(2, num_tokens=1)
        slots = self.manager.reserve_many(((1, 1), (2, 1)))
        present = []
        for layer in range(2):
            key = torch.arange(
                layer * 24,
                (layer + 1) * 24,
                dtype=torch.float32,
            ).view(2, 2, 2, 3)
            value = key + 1000
            present.append((key, value))

        self.cache.key_cache.zero_()
        self.cache.value_cache.zero_()
        self.cache.write_decode_batch(
            (slots[1], slots[2]),
            present,
            vectorized=False,
        )
        reference_key = self.cache.key_cache.clone()
        reference_value = self.cache.value_cache.clone()
        self.cache.key_cache.zero_()
        self.cache.value_cache.zero_()

        self.cache.write_decode_batch(
            (slots[1], slots[2]),
            present,
            vectorized=True,
        )
        torch.testing.assert_close(self.cache.key_cache, reference_key)
        torch.testing.assert_close(self.cache.value_cache, reference_value)


if __name__ == "__main__":
    unittest.main()

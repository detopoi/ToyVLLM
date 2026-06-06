import unittest

import torch

from toyvllm.block_manager import BlockManager
from toyvllm.kv_cache import PagedKVCache
from toyvllm.layers.attention import Qwen3Attention


class PagedAttentionTest(unittest.TestCase):
    def test_online_softmax_matches_contiguous_attention(self) -> None:
        """即使物理块不连续，分页结果也应与连续历史缓存相同。"""

        torch.manual_seed(0)
        attention = Qwen3Attention(
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            rms_norm_eps=1e-6,
            rope_theta=1_000_000.0,
        ).eval()
        manager = BlockManager(num_blocks=4, block_size=2)
        manager.allocate(10, num_tokens=2)  # 占用 block 0
        table = manager.allocate(20, num_tokens=2)  # request 20 使用 block 1
        manager.allocate(30, num_tokens=2)  # 占用 block 2
        manager.free(10)  # 空闲顺序变成 [3, 0]
        manager.reserve(20, 1)  # request 20 的第二个逻辑块映射到 block 3
        table = manager.get_block_table(20)

        cache = PagedKVCache(
            num_layers=1,
            num_blocks=4,
            block_size=2,
            num_kv_heads=2,
            head_dim=8,
            dtype=torch.float32,
            device="cpu",
        )
        history_key = torch.randn(1, 2, 3, 8)
        history_value = torch.randn(1, 2, 3, 8)
        cache.write(table.slots(), [(history_key, history_value)])

        hidden_states = torch.randn(1, 1, 32)
        position_ids = torch.tensor([[3]])
        expected, _ = attention(
            hidden_states,
            position_ids,
            past_key_value=(history_key, history_value),
            use_cache=True,
        )
        actual, present = attention(
            hidden_states,
            paged_attention=cache.attention_metadata((table,)),
            use_cache=True,
        )

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        self.assertEqual(present[0].shape, (1, 2, 1, 8))


if __name__ == "__main__":
    unittest.main()

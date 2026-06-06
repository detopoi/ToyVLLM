import unittest

import torch

from tests.test_layers import tiny_config
from toyvllm.models.qwen3 import Qwen3ForCausalLM


class Qwen3ModelTest(unittest.TestCase):
    def test_full_model_last_token_logits(self) -> None:
        config = tiny_config()
        model = Qwen3ForCausalLM(config)
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        logits = model(input_ids)
        self.assertEqual(logits.shape, (2, 1, config.vocab_size))
        self.assertTrue(torch.isfinite(logits).all())

    def test_cached_decode_matches_full_sequence(self) -> None:
        torch.manual_seed(0)
        config = tiny_config()
        model = Qwen3ForCausalLM(config).eval()

        full_input = torch.tensor([[1, 2, 3, 4]])
        full_logits = model(full_input)

        _, cache = model(
            full_input[:, :3],
            use_cache=True,
        )
        cached_logits, cache = model(
            full_input[:, 3:],
            past_key_values=cache,
            use_cache=True,
        )

        torch.testing.assert_close(
            cached_logits,
            full_logits,
            atol=1e-5,
            rtol=1e-5,
        )
        self.assertEqual(cache[0][0].shape[2], 4)
        self.assertEqual(cache[0][0].shape[1], config.num_key_value_heads)

    def test_cached_chunk_matches_full_sequence(self) -> None:
        """带历史缓存的一次多 token 输入也必须使用正确的偏移因果 mask。"""

        torch.manual_seed(0)
        model = Qwen3ForCausalLM(tiny_config()).eval()
        full_input = torch.tensor([[1, 2, 3, 4]])
        full_logits = model(full_input)

        _, cache = model(full_input[:, :2], use_cache=True)
        chunk_logits, _ = model(
            full_input[:, 2:],
            past_key_values=cache,
            use_cache=True,
        )
        torch.testing.assert_close(
            chunk_logits,
            full_logits,
            atol=1e-5,
            rtol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()

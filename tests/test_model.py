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

    def test_left_padded_batch_matches_individual_logits(self) -> None:
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(tiny_config()).eval()
        first = torch.tensor([[1, 2, 3, 4]])
        second = torch.tensor([[5, 6]])
        first_logits = model(first)
        second_logits = model(second)

        batch = torch.tensor(
            [
                [1, 2, 3, 4],
                [0, 0, 5, 6],
            ]
        )
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1],
                [0, 0, 1, 1],
            ]
        )
        batch_logits = model(batch, attention_mask=attention_mask)

        torch.testing.assert_close(batch_logits[0], first_logits[0])
        torch.testing.assert_close(
            batch_logits[1],
            second_logits[0],
            atol=1e-5,
            rtol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()

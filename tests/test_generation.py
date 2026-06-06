import unittest

import torch

from tests.test_layers import tiny_config
from toyvllm.generation import generate_greedy_cached, generate_greedy_naive
from toyvllm.models.qwen3 import Qwen3ForCausalLM


class GenerationTest(unittest.TestCase):
    def test_naive_generation_returns_requested_tokens(self) -> None:
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(tiny_config()).eval()
        result = generate_greedy_naive(
            model,
            [1, 2, 3],
            max_new_tokens=3,
            eos_token_ids=set(),
        )
        self.assertEqual(len(result.output_token_ids), 3)
        self.assertEqual(len(result.step_seconds), 3)
        self.assertGreater(result.output_tokens_per_second, 0)
        self.assertGreater(result.decode_tokens_per_second, 0)
        self.assertEqual(result.peak_memory_mib, 0.0)

    def test_cached_generation_matches_naive(self) -> None:
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(tiny_config()).eval()
        arguments = {
            "prompt_token_ids": [1, 2, 3, 4],
            "max_new_tokens": 5,
            "eos_token_ids": set(),
        }
        naive = generate_greedy_naive(model, **arguments)
        cached = generate_greedy_cached(model, **arguments)
        self.assertEqual(cached.output_token_ids, naive.output_token_ids)


if __name__ == "__main__":
    unittest.main()

import unittest

import torch

from tests.test_layers import tiny_config
from toyvllm.generation import (
    generate_greedy_cached,
    generate_greedy_naive,
    generate_static_batch,
)
from toyvllm.models.qwen3 import Qwen3ForCausalLM
from toyvllm.sampling import SamplingParams


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

    def test_sampled_generation_is_reproducible_across_backends(self) -> None:
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(tiny_config()).eval()
        arguments = {
            "prompt_token_ids": [1, 2, 3, 4],
            "max_new_tokens": 8,
            "eos_token_ids": set(),
            "sampling_params": SamplingParams(
                temperature=0.8,
                top_k=10,
                top_p=0.9,
                seed=123,
            ),
        }
        naive = generate_greedy_naive(model, **arguments)
        cached = generate_greedy_cached(model, **arguments)
        self.assertEqual(cached.output_token_ids, naive.output_token_ids)

    def test_static_batch_matches_individual_cached_generation(self) -> None:
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(tiny_config()).eval()
        prompts = [[1, 2, 3, 4], [5, 6]]
        batch = generate_static_batch(
            model,
            prompts,
            max_new_tokens=5,
            eos_token_ids=set(),
            pad_token_id=0,
        )
        individual = [
            generate_greedy_cached(
                model,
                prompt,
                max_new_tokens=5,
                eos_token_ids=set(),
            ).output_token_ids
            for prompt in prompts
        ]
        self.assertEqual(batch.output_token_ids, individual)

    def test_static_batch_first_sampling_stream_matches_single_request(self) -> None:
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(tiny_config()).eval()
        params = SamplingParams(
            temperature=0.8,
            top_k=10,
            top_p=0.9,
            seed=123,
        )
        batch = generate_static_batch(
            model,
            [[1, 2, 3], [4, 5]],
            max_new_tokens=5,
            eos_token_ids=set(),
            pad_token_id=0,
            sampling_params=params,
        )
        single = generate_greedy_cached(
            model,
            [1, 2, 3],
            max_new_tokens=5,
            eos_token_ids=set(),
            sampling_params=params,
        )
        self.assertEqual(batch.output_token_ids[0], single.output_token_ids)

    def test_static_batch_stops_finished_request_independently(self) -> None:
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(tiny_config()).eval()
        prompts = [[1, 2, 3], [4, 5]]
        first_tokens = [
            generate_greedy_cached(
                model,
                prompt,
                max_new_tokens=1,
                eos_token_ids=set(),
            ).output_token_ids[0]
            for prompt in prompts
        ]
        self.assertNotEqual(first_tokens[0], first_tokens[1])

        batch = generate_static_batch(
            model,
            prompts,
            max_new_tokens=3,
            eos_token_ids={first_tokens[0]},
            pad_token_id=0,
        )
        self.assertEqual(len(batch.output_token_ids[0]), 1)
        self.assertGreater(len(batch.output_token_ids[1]), 1)


if __name__ == "__main__":
    unittest.main()

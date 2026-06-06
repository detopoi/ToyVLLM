import unittest

import torch

from tests.test_layers import tiny_config
from toyvllm.engine import PagedContinuousBatchEngine
from toyvllm.generation import generate_greedy_cached
from toyvllm.models.qwen3 import Qwen3ForCausalLM


class PagedContinuousBatchEngineTest(unittest.TestCase):
    def test_paged_engine_matches_individual_generation(self) -> None:
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(tiny_config()).eval()
        prompts = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        limits = [2, 5, 2]
        expected = [
            generate_greedy_cached(
                model,
                prompt,
                max_new_tokens=limit,
                eos_token_ids=set(),
            ).output_token_ids
            for prompt, limit in zip(prompts, limits)
        ]

        engine = PagedContinuousBatchEngine(
            model,
            max_num_seqs=2,
            pad_token_id=0,
            num_blocks=8,
            block_size=2,
        )
        for prompt, limit in zip(prompts, limits):
            engine.add_request(
                prompt,
                max_new_tokens=limit,
                eos_token_ids=set(),
            )
        result = engine.run()

        self.assertEqual(
            [sequence.output_token_ids for sequence in result.sequences],
            expected,
        )
        self.assertEqual(
            engine.block_manager.stats.num_free_blocks,
            engine.block_manager.stats.num_total_blocks,
        )

    def test_block_capacity_delays_fifo_request_until_release(self) -> None:
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(tiny_config()).eval()
        engine = PagedContinuousBatchEngine(
            model,
            max_num_seqs=2,
            pad_token_id=0,
            num_blocks=1,
            block_size=4,
        )
        for prompt in ([1, 2], [3, 4]):
            engine.add_request(
                list(prompt),
                max_new_tokens=1,
                eos_token_ids=set(),
            )
        result = engine.run()

        self.assertEqual(result.iterations[0].prefill_request_ids, (0,))
        self.assertEqual(result.iterations[1].prefill_request_ids, (1,))
        self.assertEqual(result.sequences[1].admitted_step, 1)
        self.assertEqual(engine.block_manager.stats.num_used_blocks, 0)


if __name__ == "__main__":
    unittest.main()

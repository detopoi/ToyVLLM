import unittest

import torch

from tests.test_layers import tiny_config
from toyvllm.engine import PagedContinuousBatchEngine
from toyvllm.generation import generate_greedy_cached
from toyvllm.models.qwen3 import Qwen3ForCausalLM


class PagedContinuousBatchEngineTest(unittest.TestCase):
    def test_chunked_prefill_matches_individual_generation(self) -> None:
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(tiny_config()).eval()
        prompts = [[1, 2], [3, 4, 5, 6, 7, 8, 9]]
        limits = [4, 3]
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
            num_blocks=12,
            block_size=2,
            attention_backend="paged",
            max_num_batched_tokens=4,
            max_prefill_chunk_size=2,
            max_mixed_prefill_tokens=2,
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
        self.assertTrue(
            any(
                iteration.decode_request_ids
                and iteration.prefill_request_ids
                for iteration in result.iterations
            )
        )
        for iteration in result.iterations:
            scheduled_tokens = (
                len(iteration.decode_request_ids)
                + sum(iteration.prefill_token_counts)
            )
            self.assertLessEqual(scheduled_tokens, 4)
            if iteration.decode_request_ids:
                self.assertLessEqual(sum(iteration.prefill_token_counts), 2)
        self.assertEqual(
            engine.block_manager.stats.num_free_blocks,
            engine.block_manager.stats.num_total_blocks,
        )

    def test_chunked_prefill_batches_different_history_lengths(self) -> None:
        torch.manual_seed(1)
        model = Qwen3ForCausalLM(tiny_config()).eval()
        prompts = [
            [1, 2, 3, 4],
            [5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 15, 16, 17, 18, 19, 20],
        ]
        expected = [
            generate_greedy_cached(
                model,
                prompt,
                max_new_tokens=2,
                eos_token_ids=set(),
            ).output_token_ids
            for prompt in prompts
        ]
        engine = PagedContinuousBatchEngine(
            model,
            max_num_seqs=3,
            pad_token_id=0,
            num_blocks=16,
            block_size=2,
            attention_backend="paged",
            max_num_batched_tokens=7,
            max_prefill_chunk_size=4,
        )
        for prompt in prompts:
            engine.add_request(
                prompt,
                max_new_tokens=2,
                eos_token_ids=set(),
            )
        result = engine.run()

        self.assertEqual(
            [sequence.output_token_ids for sequence in result.sequences],
            expected,
        )
        self.assertTrue(
            any(
                len(set(iteration.prefill_token_counts)) > 1
                for iteration in result.iterations
                if len(iteration.prefill_token_counts) > 1
            )
        )

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

    def test_paged_attention_does_not_call_batch_gather(self) -> None:
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(tiny_config()).eval()
        engine = PagedContinuousBatchEngine(
            model,
            max_num_seqs=2,
            pad_token_id=0,
            num_blocks=8,
            block_size=2,
        )

        def fail_if_called(*args: object, **kwargs: object) -> None:
            raise AssertionError("9C Decode 不应调用 read_batch")

        engine.paged_cache.read_batch = fail_if_called
        engine.add_request(
            [1, 2, 3],
            max_new_tokens=3,
            eos_token_ids=set(),
        )
        result = engine.run()
        self.assertEqual(len(result.sequences[0].output_token_ids), 3)

    def test_gather_and_paged_attention_generate_same_tokens(self) -> None:
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(tiny_config()).eval()
        outputs = []
        for backend in ("gather", "paged"):
            engine = PagedContinuousBatchEngine(
                model,
                max_num_seqs=2,
                pad_token_id=0,
                num_blocks=8,
                block_size=2,
                attention_backend=backend,
            )
            for prompt in ([1, 2, 3], [4, 5]):
                engine.add_request(
                    list(prompt),
                    max_new_tokens=4,
                    eos_token_ids=set(),
                )
            outputs.append(
                [
                    sequence.output_token_ids
                    for sequence in engine.run().sequences
                ]
            )

        self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()

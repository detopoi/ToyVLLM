import unittest

import torch

from tests.test_layers import tiny_config
from toyvllm.engine import ContinuousBatchEngine
from toyvllm.generation import generate_greedy_cached
from toyvllm.models.qwen3 import Qwen3ForCausalLM


class ContinuousBatchEngineTest(unittest.TestCase):
    def test_dynamic_admission_matches_individual_generation(self) -> None:
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

        engine = ContinuousBatchEngine(
            model,
            max_num_seqs=2,
            pad_token_id=0,
        )
        for prompt, limit in zip(prompts, limits):
            engine.add_request(
                prompt,
                max_new_tokens=limit,
                eos_token_ids=set(),
            )
        result = engine.run()

        actual = [sequence.output_token_ids for sequence in result.sequences]
        self.assertEqual(actual, expected)
        self.assertEqual(result.completion_order, (0, 2, 1))

        # 第 0 轮先 prefill 请求 0/1；第 1 轮请求 0 完成，空槽立即给请求 2。
        self.assertEqual(result.iterations[0].prefill_request_ids, (0, 1))
        self.assertEqual(result.iterations[1].decode_request_ids, (0, 1))
        self.assertEqual(result.iterations[1].prefill_request_ids, (2,))
        self.assertEqual(result.sequences[2].admitted_step, 1)

    def test_max_concurrency_is_never_exceeded(self) -> None:
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(tiny_config()).eval()
        engine = ContinuousBatchEngine(
            model,
            max_num_seqs=2,
            pad_token_id=0,
        )
        for token in range(1, 6):
            engine.add_request(
                [token],
                max_new_tokens=2,
                eos_token_ids=set(),
            )
        result = engine.run()

        for iteration in result.iterations:
            self.assertLessEqual(len(iteration.decode_request_ids), 2)
            self.assertLessEqual(len(iteration.prefill_request_ids), 2)

    def test_request_can_arrive_between_steps(self) -> None:
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(tiny_config()).eval()
        engine = ContinuousBatchEngine(
            model,
            max_num_seqs=2,
            pad_token_id=0,
        )
        engine.add_request(
            [1, 2],
            max_new_tokens=3,
            eos_token_ids=set(),
        )
        first_step = engine.step()
        self.assertEqual(first_step.prefill_request_ids, (0,))

        # 模拟服务运行期间新请求到达：先进入 waiting，下一轮获得空槽。
        engine.add_request(
            [3, 4],
            max_new_tokens=2,
            eos_token_ids=set(),
        )
        second_step = engine.step()
        self.assertEqual(second_step.decode_request_ids, (0,))
        self.assertEqual(second_step.prefill_request_ids, (1,))

        result = engine.run()
        self.assertEqual(len(result.sequences), 2)
        self.assertTrue(
            all(
                sequence.finished_step is not None
                for sequence in result.sequences
            )
        )


if __name__ == "__main__":
    unittest.main()

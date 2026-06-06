import unittest

import torch

from toyvllm.sampling import (
    SamplingParams,
    create_generator,
    create_generators,
    filter_logits,
    sample_next_token,
)


class SamplingTest(unittest.TestCase):
    def test_greedy_selects_largest_logit(self) -> None:
        logits = torch.tensor([[1.0, 3.0, 2.0]])
        token = sample_next_token(logits, SamplingParams())
        self.assertEqual(token.item(), 1)

    def test_top_k_keeps_exactly_k_candidates(self) -> None:
        logits = torch.tensor([[1.0, 4.0, 3.0, 2.0]])
        filtered = filter_logits(
            logits,
            SamplingParams(temperature=1.0, top_k=2),
        )
        kept = torch.isfinite(filtered).nonzero(as_tuple=False)[:, 1].tolist()
        self.assertEqual(kept, [1, 2])

    def test_top_p_keeps_minimum_nucleus(self) -> None:
        # softmax 后概率约为 [0.64, 0.24, 0.09, 0.03]。
        logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
        filtered = filter_logits(
            logits,
            SamplingParams(temperature=1.0, top_p=0.8),
        )
        kept = torch.isfinite(filtered).nonzero(as_tuple=False)[:, 1].tolist()
        self.assertEqual(kept, [0, 1])

    def test_same_seed_reproduces_sequence(self) -> None:
        params = SamplingParams(temperature=0.8, top_k=3, seed=123)
        logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])

        first_generator = create_generator(params, logits.device)
        second_generator = create_generator(params, logits.device)
        first = [
            sample_next_token(logits, params, generator=first_generator).item()
            for _ in range(20)
        ]
        second = [
            sample_next_token(logits, params, generator=second_generator).item()
            for _ in range(20)
        ]
        self.assertEqual(first, second)

    def test_invalid_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SamplingParams(temperature=-1)
        with self.assertRaises(ValueError):
            SamplingParams(top_k=-1)
        with self.assertRaises(ValueError):
            SamplingParams(top_p=0)

    def test_batch_generators_use_independent_reproducible_seeds(self) -> None:
        params = SamplingParams(temperature=1.0, seed=10)
        generators = create_generators(params, torch.device("cpu"), 2)
        first = torch.rand(3, generator=generators[0])
        second = torch.rand(3, generator=generators[1])

        repeated = create_generators(params, torch.device("cpu"), 2)
        torch.testing.assert_close(
            first,
            torch.rand(3, generator=repeated[0]),
        )
        self.assertFalse(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()

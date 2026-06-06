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


if __name__ == "__main__":
    unittest.main()

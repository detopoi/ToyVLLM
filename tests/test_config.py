import unittest

from toyvllm.config import ModelConfig


class ModelConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = ModelConfig.from_pretrained("Qwen3-1.7B")

    def test_qwen3_dimensions(self) -> None:
        self.assertEqual(self.config.hidden_size, 2048)
        self.assertEqual(self.config.num_hidden_layers, 28)
        self.assertEqual(self.config.queries_per_kv, 2)
        self.assertEqual(self.config.generation_eos_token_ids, (151645, 151643))
        self.assertTrue(self.config.generation_do_sample)
        self.assertEqual(self.config.generation_temperature, 0.6)
        self.assertEqual(self.config.generation_top_k, 20)
        self.assertEqual(self.config.generation_top_p, 0.95)

    def test_kv_cache_size(self) -> None:
        self.assertEqual(self.config.kv_cache_bytes_per_token, 114688)
        self.assertEqual(self.config.kv_cache_mib(4096), 448.0)


if __name__ == "__main__":
    unittest.main()

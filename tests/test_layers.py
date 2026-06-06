import unittest
from pathlib import Path

import torch

from toyvllm.config import ModelConfig
from toyvllm.layers.attention import Qwen3Attention
from toyvllm.layers.mlp import Qwen3MLP
from toyvllm.layers.rms_norm import RMSNorm
from toyvllm.layers.rotary_embedding import RotaryEmbedding
from toyvllm.models.qwen3 import Qwen3DecoderLayer


def tiny_config() -> ModelConfig:
    return ModelConfig(
        model_path=Path("."),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=128,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        torch_dtype="float32",
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
    )


class RMSNormTest(unittest.TestCase):
    def test_matches_formula(self) -> None:
        layer = RMSNorm(4, eps=1e-6)
        inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        expected = inputs * torch.rsqrt(inputs.pow(2).mean(-1, keepdim=True) + 1e-6)
        torch.testing.assert_close(layer(inputs), expected)


class RotaryEmbeddingTest(unittest.TestCase):
    def test_position_zero_is_unchanged_and_norm_is_preserved(self) -> None:
        torch.manual_seed(0)
        rope = RotaryEmbedding(head_dim=8)
        query = torch.randn(1, 4, 3, 8)
        key = torch.randn(1, 2, 3, 8)
        positions = torch.arange(3).unsqueeze(0)
        rotated_query, rotated_key = rope(query, key, positions)

        torch.testing.assert_close(rotated_query[:, :, 0], query[:, :, 0])
        torch.testing.assert_close(rotated_key[:, :, 0], key[:, :, 0])
        torch.testing.assert_close(
            rotated_query.norm(dim=-1),
            query.norm(dim=-1),
        )


class Qwen3LayerTest(unittest.TestCase):
    def test_mlp_shape(self) -> None:
        layer = Qwen3MLP(hidden_size=32, intermediate_size=64)
        output = layer(torch.randn(2, 5, 32))
        self.assertEqual(output.shape, (2, 5, 32))

    def test_attention_is_causal(self) -> None:
        torch.manual_seed(0)
        layer = Qwen3Attention(
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            rms_norm_eps=1e-6,
            rope_theta=1_000_000.0,
        )
        inputs = torch.randn(1, 4, 32)
        changed_future = inputs.clone()
        changed_future[:, 2:] = torch.randn_like(changed_future[:, 2:])

        original_output = layer(inputs)
        changed_output = layer(changed_future)

        # 未来 token 被替换后，位置 0 和 1 的结果不能变化，否则说明看到了未来。
        torch.testing.assert_close(
            original_output[:, :2],
            changed_output[:, :2],
            atol=1e-5,
            rtol=1e-5,
        )

    def test_decoder_layer_shape_and_finite_values(self) -> None:
        layer = Qwen3DecoderLayer(tiny_config())
        output = layer(torch.randn(2, 5, 32))
        self.assertEqual(output.shape, (2, 5, 32))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()

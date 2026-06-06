import unittest

import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer as TransformersQwen3DecoderLayer,
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from tests.test_layers import tiny_config
from toyvllm.models.qwen3 import Qwen3DecoderLayer


class TransformersReferenceTest(unittest.TestCase):
    def test_decoder_layer_matches_transformers(self) -> None:
        """同权重、同输入时，应与官方 Qwen3 层得到几乎相同的输出。"""

        torch.manual_seed(0)
        config = tiny_config()
        ours = Qwen3DecoderLayer(config).eval()

        reference_config = Qwen3Config(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            attention_bias=False,
            attention_dropout=0.0,
        )
        reference_config._attn_implementation = "eager"
        reference = TransformersQwen3DecoderLayer(reference_config, layer_idx=0)
        reference.load_state_dict(ours.state_dict(), strict=True)
        reference.eval()

        hidden_states = torch.randn(2, 5, config.hidden_size)
        position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)
        rotary = Qwen3RotaryEmbedding(reference_config)
        position_embeddings = rotary(hidden_states, position_ids)

        # Transformers 的 eager attention 接收加法 mask：允许的位置加 0，
        # 未来位置加 -inf，softmax 后未来位置的概率就会变为 0。
        attention_mask = torch.full((2, 1, 5, 5), float("-inf"))
        attention_mask = torch.triu(attention_mask, diagonal=1)

        with torch.no_grad():
            actual = ours(hidden_states, position_ids)
            expected = reference(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0]

        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()

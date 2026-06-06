from __future__ import annotations

import torch
from torch import nn

from toyvllm.config import ModelConfig
from toyvllm.layers.attention import Qwen3Attention
from toyvllm.layers.mlp import Qwen3MLP
from toyvllm.layers.rms_norm import RMSNorm


class Qwen3DecoderLayer(nn.Module):
    """一个完整 Qwen3 Decoder Layer：Attention + MLP + 两次残差连接。"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.mlp = Qwen3MLP(
            config.hidden_size,
            config.intermediate_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-Norm 结构先归一化再进入子层。残差支路保留原信息，也让深层网络
        # 的信息和梯度有一条直接通道。
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states, position_ids)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class Qwen3Model(nn.Module):
    """不含语言模型输出头的 Qwen3 Transformer 主体。"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids 的形状必须是 [batch, sequence]")

        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids)
        return self.norm(hidden_states)


class Qwen3ForCausalLM(nn.Module):
    """完整 Qwen3 因果语言模型：Transformer 主体加词表输出头。"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        last_token_only: bool = True,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids)

        # 生成下一个 token 时只需要最后一个位置的 logits。若对整段序列都投影到
        # 151936 维词表，会制造一个很大的临时张量，却不会改变采样结果。
        if last_token_only:
            hidden_states = hidden_states[:, -1:, :]
        return self.lm_head(hidden_states)

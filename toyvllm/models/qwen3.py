from __future__ import annotations

import torch
from torch import nn

from toyvllm.config import ModelConfig
from toyvllm.layers.attention import KVCache, Qwen3Attention
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
        attention_mask: torch.Tensor | None = None,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        # Pre-Norm 结构先归一化再进入子层。残差支路保留原信息，也让深层网络
        # 的信息和梯度有一条直接通道。
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_output = self.self_attn(
            hidden_states,
            position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        present_key_value = None
        if use_cache:
            attention_output, present_key_value = attention_output
        hidden_states = residual + attention_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        if use_cache:
            if present_key_value is None:
                raise RuntimeError("use_cache=True 时 Attention 没有返回缓存")
            return hidden_states, present_key_value
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
        attention_mask: torch.Tensor | None = None,
        past_key_values: list[KVCache] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[KVCache]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids 的形状必须是 [batch, sequence]")
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError("past_key_values 的层数与模型层数不一致")

        past_length = 0
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]
        if position_ids is None:
            if attention_mask is not None:
                # 左 padding 不应占用 RoPE 位置。真实 token 的位置始终从 0 开始。
                all_position_ids = attention_mask.long().cumsum(dim=-1) - 1
                all_position_ids.clamp_(min=0)
                position_ids = all_position_ids[:, -input_ids.shape[1] :]
            else:
                sequence_length = input_ids.shape[1]
                position_ids = torch.arange(
                    past_length,
                    past_length + sequence_length,
                    device=input_ids.device,
                ).unsqueeze(0).expand(input_ids.shape[0], -1)

        hidden_states = self.embed_tokens(input_ids)
        present_key_values: list[KVCache] = []
        for layer_index, layer in enumerate(self.layers):
            layer_past = (
                None if past_key_values is None else past_key_values[layer_index]
            )
            layer_output = layer(
                hidden_states,
                position_ids,
                attention_mask=attention_mask,
                past_key_value=layer_past,
                use_cache=use_cache,
            )
            if use_cache:
                hidden_states, layer_present = layer_output
                present_key_values.append(layer_present)
            else:
                hidden_states = layer_output

        hidden_states = self.norm(hidden_states)
        if use_cache:
            return hidden_states, present_key_values
        return hidden_states


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
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list[KVCache] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[KVCache]]:
        model_output = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        present_key_values = None
        if use_cache:
            hidden_states, present_key_values = model_output
        else:
            hidden_states = model_output

        # 生成下一个 token 时只需要最后一个位置的 logits。若对整段序列都投影到
        # 151936 维词表，会制造一个很大的临时张量，却不会改变采样结果。
        if last_token_only:
            hidden_states = hidden_states[:, -1:, :]
        logits = self.lm_head(hidden_states)
        if use_cache:
            if present_key_values is None:
                raise RuntimeError("use_cache=True 时模型没有返回缓存")
            return logits, present_key_values
        return logits

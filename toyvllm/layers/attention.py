from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from toyvllm.layers.rms_norm import RMSNorm
from toyvllm.layers.rotary_embedding import RotaryEmbedding


class Qwen3Attention(nn.Module):
    """Qwen3 的 Grouped Query Attention（当前为无 KV Cache 版本）。"""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        rope_theta: float,
    ) -> None:
        super().__init__()
        if num_attention_heads * head_dim != hidden_size:
            raise ValueError("num_attention_heads * head_dim 必须等于 hidden_size")
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError("Query Head 数必须能被 KV Head 数整除")

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.queries_per_kv = num_attention_heads // num_key_value_heads

        self.q_proj = nn.Linear(
            hidden_size,
            num_attention_heads * head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            hidden_size,
            num_key_value_heads * head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            hidden_size,
            num_key_value_heads * head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            num_attention_heads * head_dim,
            hidden_size,
            bias=False,
        )

        # Qwen3 比普通 Llama 多了每个注意力头内部的 Q/K RMSNorm。
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.rotary_embedding = RotaryEmbedding(head_dim, theta=rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        if position_ids is None:
            position_ids = torch.arange(
                sequence_length,
                device=hidden_states.device,
            ).unsqueeze(0).expand(batch_size, -1)

        # 投影后先拆成多个头，再把 head 维移到 sequence 前面：
        # [batch, sequence, heads * head_dim]
        # -> [batch, heads, sequence, head_dim]
        query = self._split_heads(
            self.q_proj(hidden_states),
            self.num_attention_heads,
        )
        key = self._split_heads(
            self.k_proj(hidden_states),
            self.num_key_value_heads,
        )
        value = self._split_heads(
            self.v_proj(hidden_states),
            self.num_key_value_heads,
        )

        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = self.rotary_embedding(query, key, position_ids)

        # 当前 PyTorch 2.2.2 的 SDPA 没有直接启用 GQA 的参数，因此先显式复制
        # K/V Head，让它们和 Query Head 数一致。逻辑正确但会增加临时显存，
        # 后续优化时会把这里作为一个明确的 benchmark 对象。
        key = key.repeat_interleave(self.queries_per_kv, dim=1)
        value = value.repeat_interleave(self.queries_per_kv, dim=1)

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=True,
        )

        # [batch, heads, sequence, head_dim]
        # -> [batch, sequence, heads * head_dim]
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, sequence_length, self.hidden_size)
        return self.o_proj(attended)

    def _split_heads(self, tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
        batch_size, sequence_length, _ = tensor.shape
        return tensor.view(
            batch_size,
            sequence_length,
            num_heads,
            self.head_dim,
        ).transpose(1, 2)


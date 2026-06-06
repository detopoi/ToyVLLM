from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from toyvllm.layers.rms_norm import RMSNorm
from toyvllm.layers.rotary_embedding import RotaryEmbedding

KVCache = tuple[torch.Tensor, torch.Tensor]


class Qwen3Attention(nn.Module):
    """Qwen3 的 Grouped Query Attention，支持连续 KV Cache。"""

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
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        batch_size, sequence_length, _ = hidden_states.shape
        past_length = 0 if past_key_value is None else past_key_value[0].shape[2]
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                past_length + sequence_length,
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

        if past_key_value is not None:
            past_key, past_value = past_key_value
            # 连续 KV Cache 沿 token 维追加。本阶段先使用 cat 保持实现直观；
            # 它每步都会重新分配连续空间，Paged KV Cache 阶段会解决这个问题。
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)
        present_key_value = (key, value)

        # 当前 PyTorch 2.2.2 的 SDPA 没有直接启用 GQA 的参数，因此先显式复制
        # K/V Head，让它们和 Query Head 数一致。逻辑正确但会增加临时显存，
        # 后续优化时会把这里作为一个明确的 benchmark 对象。
        key = key.repeat_interleave(self.queries_per_kv, dim=1)
        value = value.repeat_interleave(self.queries_per_kv, dim=1)

        # 没有历史缓存的 prefill 可以使用 SDPA 内置因果掩码。带缓存时 query 的
        # 第 0 个位置实际对应 past_length，而不是整段序列的位置 0，需要显式
        # 构造带偏移的 mask。单 token decode 位于序列末尾，可以看到全部历史。
        attention_mask = None
        is_causal = past_key_value is None and sequence_length > 1
        if past_key_value is not None and sequence_length > 1:
            key_positions = torch.arange(key.shape[2], device=key.device)
            query_positions = torch.arange(
                past_length,
                past_length + sequence_length,
                device=query.device,
            )
            attention_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )

        # [batch, heads, sequence, head_dim]
        # -> [batch, sequence, heads * head_dim]
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, sequence_length, self.hidden_size)
        output = self.o_proj(attended)
        if use_cache:
            return output, present_key_value
        return output

    def _split_heads(self, tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
        batch_size, sequence_length, _ = tensor.shape
        return tensor.view(
            batch_size,
            sequence_length,
            num_heads,
            self.head_dim,
        ).transpose(1, 2)

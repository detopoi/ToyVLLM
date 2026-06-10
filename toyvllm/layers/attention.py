from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from toyvllm.engine.block_manager import BlockTable
from toyvllm.layers.rms_norm import RMSNorm
from toyvllm.layers.rotary_embedding import RotaryEmbedding

KVCache = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class PagedAttentionMetadata:
    """Paged Attention 解码一次所需的只读元数据。

    Key/Value 物理池仍然是 GPU Tensor；BlockTable 很小，负责描述每条请求的逻辑块
    映射到了哪些物理块。Attention 根据 layer_index 只读取当前层的数据。
    """

    key_cache: torch.Tensor
    value_cache: torch.Tensor
    block_tables: tuple[BlockTable, ...]
    backend: str = "paged"
    block_table_tensor: torch.Tensor | None = None
    context_lengths: torch.Tensor | None = None


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
        layer_index: int = 0,
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
        self.layer_index = layer_index

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
        attention_mask: torch.Tensor | None = None,
        past_key_value: KVCache | None = None,
        paged_attention: PagedAttentionMetadata | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        batch_size, sequence_length, _ = hidden_states.shape
        if paged_attention is not None:
            if past_key_value is not None:
                raise ValueError("Paged Attention 不能同时接收连续 past_key_value")
            if sequence_length != 1:
                raise ValueError("当前 Paged Attention 只处理单 token Decode")
            if attention_mask is not None:
                raise ValueError("Paged Attention 通过 BlockTable 确定有效历史，无需 mask")
            if len(paged_attention.block_tables) != batch_size:
                raise ValueError("BlockTable 数量必须等于 batch size")

        past_length = 0 if past_key_value is None else past_key_value[0].shape[2]
        if position_ids is None:
            if paged_attention is not None:
                # 每条请求的历史长度可能不同，不能再用一个 batch 共享的 past_length。
                position_ids = torch.tensor(
                    [
                        [table.num_tokens]
                        for table in paged_attention.block_tables
                    ],
                    dtype=torch.long,
                    device=hidden_states.device,
                )
            else:
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

        if paged_attention is not None:
            # 分页路径不会把历史 K/V 与当前 K/V 做 torch.cat。历史数据留在物理块池，
            # 当前 token 的 K/V 一边参与本轮 Attention，一边作为 present 返回给 Engine，
            # 由 Engine 在模型前向结束后写入刚刚预留的物理槽位。
            present_key_value = (key, value)
            if paged_attention.backend in {
                "triton",
                "triton-fixed",
                "triton-grouped",
            }:
                attended = self._triton_paged_decode(
                    query,
                    key,
                    value,
                    paged_attention,
                )
            elif paged_attention.backend == "paged":
                attended = self._paged_decode(query, key, value, paged_attention)
            else:
                raise ValueError(
                    f"未知 Paged Attention 后端：{paged_attention.backend}"
                )
            attended = attended.transpose(1, 2).contiguous()
            attended = attended.view(batch_size, sequence_length, self.hidden_size)
            output = self.o_proj(attended)
            if use_cache:
                return output, present_key_value
            return output

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
        sdpa_mask = None
        has_padding = attention_mask is not None
        is_causal = (
            past_key_value is None
            and sequence_length > 1
            and not has_padding
        )
        if has_padding:
            if attention_mask.ndim != 2:
                raise ValueError("attention_mask 的形状必须是 [batch, key_sequence]")
            if attention_mask.shape != (batch_size, key.shape[2]):
                raise ValueError(
                    "attention_mask 长度必须覆盖历史缓存和当前输入"
                )

            key_is_valid = attention_mask.to(torch.bool)[:, None, None, :]
            key_positions = torch.arange(key.shape[2], device=key.device)
            query_positions = torch.arange(
                past_length,
                past_length + sequence_length,
                device=query.device,
            )
            causal = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            sdpa_mask = key_is_valid & causal[None, None, :, :]

            # 左 padding 的 query 本身也处在无效位置，正常结果不会被读取。但如果
            # 一整行 key 都被屏蔽，某些 SDPA 后端会返回 NaN，并通过残差污染后续层。
            # 因此只让无效 query 看见自己的 padding key；真实 query 仍看不见任何 pad。
            query_is_valid = attention_mask[
                :, past_length : past_length + sequence_length
            ].to(torch.bool)
            self_only = (
                key_positions.unsqueeze(0) == query_positions.unsqueeze(1)
            )[None, None, :, :]
            sdpa_mask = torch.where(
                query_is_valid[:, None, :, None],
                sdpa_mask,
                self_only,
            )
        elif past_key_value is not None and sequence_length > 1:
            key_positions = torch.arange(key.shape[2], device=key.device)
            query_positions = torch.arange(
                past_length,
                past_length + sequence_length,
                device=query.device,
            )
            sdpa_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            sdpa_mask = sdpa_mask.unsqueeze(0).unsqueeze(0)

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=sdpa_mask,
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

    def _paged_decode(
        self,
        query: torch.Tensor,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
        metadata: PagedAttentionMetadata,
    ) -> torch.Tensor:
        """逐物理块执行单 token Decode Attention。

        普通 Attention 会先得到完整 scores，再一次 softmax。分页实现不能拼出完整历史，
        因此维护三个在线状态：

        - m：已经扫描部分的最大 score；
        - l：以 m 为基准的 exp(score) 之和；
        - acc：同一归一化基准下的加权 Value 之和。

        每读一个物理块就更新这三个状态。扫描结束后 acc / l 与对完整历史做 softmax
        数学等价，但中间张量只与 block_size 成正比，不与上下文总长度成正比。
        """

        if metadata.key_cache.shape != metadata.value_cache.shape:
            raise ValueError("Paged Key/Value 物理池形状必须一致")
        if not 0 <= self.layer_index < metadata.key_cache.shape[0]:
            raise ValueError("layer_index 超出 Paged KV Cache 层数")

        batch_outputs: list[torch.Tensor] = []
        scale = self.head_dim**-0.5
        for batch_index, table in enumerate(metadata.block_tables):
            if table.block_size != metadata.key_cache.shape[2]:
                raise ValueError("BlockTable 与 Paged KV Cache 的 block_size 不一致")

            # GQA 不需要真的复制 K/V Head。把 Query 看成
            # [kv_heads, queries_per_kv, head_dim]，同一个 KV Head 直接服务一组 Query。
            request_query = query[batch_index, :, 0].reshape(
                self.num_key_value_heads,
                self.queries_per_kv,
                self.head_dim,
            )
            running_max = torch.full(
                (self.num_key_value_heads, self.queries_per_kv),
                float("-inf"),
                dtype=torch.float32,
                device=query.device,
            )
            running_sum = torch.zeros_like(running_max)
            running_value = torch.zeros(
                (
                    self.num_key_value_heads,
                    self.queries_per_kv,
                    self.head_dim,
                ),
                dtype=torch.float32,
                device=query.device,
            )

            # 历史 token 按逻辑块顺序读取。物理块编号可以完全不连续。
            remaining = table.num_tokens
            for physical_block_id in table.physical_block_ids:
                valid_tokens = min(remaining, table.block_size)
                if valid_tokens <= 0:
                    break
                block_key = metadata.key_cache[
                    self.layer_index,
                    physical_block_id,
                    :valid_tokens,
                ].transpose(0, 1)
                block_value = metadata.value_cache[
                    self.layer_index,
                    physical_block_id,
                    :valid_tokens,
                ].transpose(0, 1)
                running_max, running_sum, running_value = self._merge_kv_block(
                    request_query,
                    block_key,
                    block_value,
                    running_max,
                    running_sum,
                    running_value,
                    scale,
                )
                remaining -= valid_tokens

            # 当前 token 尚未写入物理池，但因果 Attention 必须能看到自己，所以把它当成
            # 最后一个长度为 1 的临时块参与在线 softmax。
            token_key = current_key[batch_index, :, 0].unsqueeze(1)
            token_value = current_value[batch_index, :, 0].unsqueeze(1)
            running_max, running_sum, running_value = self._merge_kv_block(
                request_query,
                token_key,
                token_value,
                running_max,
                running_sum,
                running_value,
                scale,
            )
            output = (running_value / running_sum.unsqueeze(-1)).reshape(
                self.num_attention_heads,
                self.head_dim,
            )
            batch_outputs.append(output.to(query.dtype).unsqueeze(1))

        return torch.stack(batch_outputs, dim=0)

    def _triton_paged_decode(
        self,
        query: torch.Tensor,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
        metadata: PagedAttentionMetadata,
    ) -> torch.Tensor:
        """调用融合了 Block 扫描和在线 softmax 的 Triton Kernel。"""

        if metadata.block_table_tensor is None:
            raise ValueError("Triton 后端缺少 GPU BlockTable Tensor")
        if metadata.context_lengths is None:
            raise ValueError("Triton 后端缺少 Context Length Tensor")

        # 延迟导入让 CPU 单测和未安装 Triton 的环境仍可使用普通后端。
        from toyvllm.kernels.paged_attention import triton_paged_attention

        return triton_paged_attention(
            query,
            current_key,
            current_value,
            metadata.key_cache,
            metadata.value_cache,
            metadata.block_table_tensor,
            metadata.context_lengths,
            layer_index=self.layer_index,
            queries_per_kv=self.queries_per_kv,
            grouped=metadata.backend == "triton-grouped",
            autotune=metadata.backend == "triton",
        )

    @staticmethod
    def _merge_kv_block(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        running_max: torch.Tensor,
        running_sum: torch.Tensor,
        running_value: torch.Tensor,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """把一个 KV Block 合并进在线 softmax 状态。"""

        scores = torch.einsum(
            "hgd,htd->hgt",
            query.float(),
            key.float(),
        ) * scale
        block_max = scores.amax(dim=-1)
        new_max = torch.maximum(running_max, block_max)
        old_scale = torch.exp(running_max - new_max)
        block_weights = torch.exp(scores - new_max.unsqueeze(-1))
        new_sum = running_sum * old_scale + block_weights.sum(dim=-1)
        new_value = (
            running_value * old_scale.unsqueeze(-1)
            + torch.einsum("hgt,htd->hgd", block_weights, value.float())
        )
        return new_max, new_sum, new_value

    def _split_heads(self, tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
        batch_size, sequence_length, _ = tensor.shape
        return tensor.view(
            batch_size,
            sequence_length,
            num_heads,
            self.head_dim,
        ).transpose(1, 2)

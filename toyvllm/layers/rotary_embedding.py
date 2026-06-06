from __future__ import annotations

import torch
from torch import nn


def rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    """把向量两半组成二维旋转所需的 (-y, x)。"""

    first_half, second_half = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second_half, first_half), dim=-1)


class RotaryEmbedding(nn.Module):
    """RoPE：把 token 的绝对位置编码成 Query/Key 的旋转角度。"""

    def __init__(self, head_dim: int, theta: float = 1_000_000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE 要求 head_dim 是偶数")

        self.head_dim = head_dim
        self.theta = theta
        self._set_inv_freq()

    def _set_inv_freq(self, device: torch.device | str | None = None) -> None:
        # 不同维度使用不同旋转频率：低维转得快，高维转得慢。
        # inv_freq 不是可训练参数，但作为 buffer 会自动跟随模块移动到 GPU。
        dimension_ids = torch.arange(
            0,
            self.head_dim,
            2,
            dtype=torch.float32,
            device=device,
        )
        inv_freq = 1.0 / (self.theta ** (dimension_ids / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def materialize(self, device: torch.device | str) -> None:
        """meta 初始化后，在真实设备上重建不属于权重文件的频率表。"""

        self._set_inv_freq(device)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """对 Query 和 Key 应用相同位置旋转。

        query: [batch, query_heads, sequence, head_dim]
        key:   [batch, kv_heads, sequence, head_dim]
        position_ids: [batch, sequence]
        """

        if position_ids.ndim != 2:
            raise ValueError("position_ids 的形状必须是 [batch, sequence]")

        # [batch, sequence, head_dim / 2]
        frequencies = torch.einsum(
            "bs,d->bsd",
            position_ids.float(),
            self.inv_freq.float(),
        )
        # 前后两半使用同一组角度，配合 rotate_half 完成二维旋转。
        angles = torch.cat((frequencies, frequencies), dim=-1)
        cos = angles.cos().unsqueeze(1).to(query.dtype)
        sin = angles.sin().unsqueeze(1).to(query.dtype)

        rotated_query = query * cos + rotate_half(query) * sin
        rotated_key = key * cos + rotate_half(key) * sin
        return rotated_query, rotated_key

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Qwen3MLP(nn.Module):
    """Qwen3 使用的 SwiGLU 前馈网络。"""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gate 分支经过 SiLU 决定哪些信息通过，up 分支携带待传递的内容。
        # 两者逐元素相乘后，再投影回 hidden_size。
        gated = F.silu(self.gate_proj(x))
        values = self.up_proj(x)
        return self.down_proj(gated * values)


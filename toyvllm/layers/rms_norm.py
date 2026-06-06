from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """只根据均方根缩放输入，不减去均值。

    LayerNorm 会先减均值再除标准差；RMSNorm 省略减均值，计算更简单。
    Qwen3 在每个 Attention 和 MLP 前都使用它稳定隐藏状态的数值范围。
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype

        # BF16/FP16 的有效精度有限。归一化统计量用 FP32 计算，再转回原类型，
        # 可以减少长序列推理中的累计数值误差。
        states_fp32 = hidden_states.float()
        variance = states_fp32.pow(2).mean(dim=-1, keepdim=True)
        normalized = states_fp32 * torch.rsqrt(variance + self.eps)
        return (self.weight * normalized).to(input_dtype)


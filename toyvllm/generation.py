from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from toyvllm.models.qwen3 import Qwen3ForCausalLM


@dataclass(frozen=True)
class GenerationResult:
    output_token_ids: list[int]
    step_seconds: list[float]
    peak_memory_mib: float

    @property
    def ttft_ms(self) -> float:
        return self.step_seconds[0] * 1000

    @property
    def tpot_ms(self) -> float:
        if len(self.step_seconds) <= 1:
            return 0.0
        return sum(self.step_seconds[1:]) / (len(self.step_seconds) - 1) * 1000

    @property
    def output_tokens_per_second(self) -> float:
        return len(self.output_token_ids) / sum(self.step_seconds)


@torch.inference_mode()
def generate_greedy_naive(
    model: Qwen3ForCausalLM,
    prompt_token_ids: list[int],
    *,
    max_new_tokens: int,
    eos_token_ids: set[int],
) -> GenerationResult:
    """最朴素生成：每产生一个 token，都重新计算完整序列。"""

    if not prompt_token_ids:
        raise ValueError("prompt_token_ids 不能为空")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens 必须大于 0")

    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)
    output_token_ids: list[int] = []
    step_seconds: list[float] = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(max_new_tokens):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()

        logits = model(input_ids, last_token_only=True)
        next_token = logits[:, -1].argmax(dim=-1)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_seconds.append(time.perf_counter() - started)

        token_id = int(next_token.item())
        output_token_ids.append(token_id)
        input_ids = torch.cat((input_ids, next_token[:, None]), dim=1)
        if token_id in eos_token_ids:
            break

    peak_memory_mib = 0.0
    if device.type == "cuda":
        peak_memory_mib = torch.cuda.max_memory_allocated(device) / 1024**2

    return GenerationResult(
        output_token_ids=output_token_ids,
        step_seconds=step_seconds,
        peak_memory_mib=peak_memory_mib,
    )


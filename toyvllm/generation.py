from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from toyvllm.models.qwen3 import Qwen3ForCausalLM
from toyvllm.sampling import SamplingParams, create_generator, sample_next_token


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

    @property
    def decode_tokens_per_second(self) -> float:
        """排除首 token 的 prefill 时间，只衡量后续 decode。"""

        if len(self.step_seconds) <= 1:
            return 0.0
        return (len(self.step_seconds) - 1) / sum(self.step_seconds[1:])


@torch.inference_mode()
def generate_greedy_naive(
    model: Qwen3ForCausalLM,
    prompt_token_ids: list[int],
    *,
    max_new_tokens: int,
    eos_token_ids: set[int],
    sampling_params: SamplingParams | None = None,
) -> GenerationResult:
    """最朴素生成：每产生一个 token，都重新计算完整序列。"""

    if not prompt_token_ids:
        raise ValueError("prompt_token_ids 不能为空")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens 必须大于 0")

    device = next(model.parameters()).device
    params = sampling_params or SamplingParams()
    generator = create_generator(params, device)
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
        next_token = sample_next_token(
            logits[:, -1],
            params,
            generator=generator,
        )

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


@torch.inference_mode()
def generate_greedy_cached(
    model: Qwen3ForCausalLM,
    prompt_token_ids: list[int],
    *,
    max_new_tokens: int,
    eos_token_ids: set[int],
    sampling_params: SamplingParams | None = None,
) -> GenerationResult:
    """使用连续 KV Cache：prefill 处理 prompt，decode 每次只处理一个 token。"""

    if not prompt_token_ids:
        raise ValueError("prompt_token_ids 不能为空")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens 必须大于 0")

    device = next(model.parameters()).device
    params = sampling_params or SamplingParams()
    generator = create_generator(params, device)
    current_input = torch.tensor(
        [prompt_token_ids],
        dtype=torch.long,
        device=device,
    )
    past_key_values = None
    output_token_ids: list[int] = []
    step_seconds: list[float] = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(max_new_tokens):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()

        logits, past_key_values = model(
            current_input,
            last_token_only=True,
            past_key_values=past_key_values,
            use_cache=True,
        )
        next_token = sample_next_token(
            logits[:, -1],
            params,
            generator=generator,
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_seconds.append(time.perf_counter() - started)

        token_id = int(next_token.item())
        output_token_ids.append(token_id)
        if token_id in eos_token_ids:
            break

        # 第一次循环后 prompt 已经写入 KV Cache。此后只把刚生成的一个 token
        # 交给模型，历史 token 通过 past_key_values 提供，不再重复前向。
        current_input = next_token[:, None]

    peak_memory_mib = 0.0
    if device.type == "cuda":
        peak_memory_mib = torch.cuda.max_memory_allocated(device) / 1024**2

    return GenerationResult(
        output_token_ids=output_token_ids,
        step_seconds=step_seconds,
        peak_memory_mib=peak_memory_mib,
    )

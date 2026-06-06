from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from toyvllm.models.qwen3 import Qwen3ForCausalLM
from toyvllm.sampling import (
    SamplingParams,
    create_generator,
    create_generators,
    sample_batch,
    sample_next_token,
)


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


@dataclass(frozen=True)
class BatchGenerationResult:
    output_token_ids: list[list[int]]
    step_seconds: list[float]
    first_token_seconds: float
    peak_memory_mib: float

    @property
    def total_output_tokens(self) -> int:
        return sum(len(token_ids) for token_ids in self.output_token_ids)

    @property
    def output_tokens_per_second(self) -> float:
        return self.total_output_tokens / sum(self.step_seconds)

    @property
    def ttft_ms(self) -> float:
        return self.first_token_seconds * 1000

    @property
    def tpot_ms(self) -> float:
        if len(self.step_seconds) <= 1:
            return 0.0
        return sum(self.step_seconds[1:]) / (len(self.step_seconds) - 1) * 1000


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


@torch.inference_mode()
def generate_static_batch(
    model: Qwen3ForCausalLM,
    prompt_token_ids: list[list[int]],
    *,
    max_new_tokens: int,
    eos_token_ids: set[int],
    pad_token_id: int,
    sampling_params: SamplingParams | None = None,
) -> BatchGenerationResult:
    """固定 batch 的 KV Cache 生成，支持不同长度 prompt。"""

    if not prompt_token_ids or any(not prompt for prompt in prompt_token_ids):
        raise ValueError("batch 和每条 prompt 都不能为空")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens 必须大于 0")

    device = next(model.parameters()).device
    params = sampling_params or SamplingParams()
    batch_size = len(prompt_token_ids)
    generators = create_generators(params, device, batch_size)
    max_prompt_length = max(len(prompt) for prompt in prompt_token_ids)

    padded_inputs = []
    masks = []
    for prompt in prompt_token_ids:
        padding_length = max_prompt_length - len(prompt)
        padded_inputs.append([pad_token_id] * padding_length + prompt)
        masks.append([0] * padding_length + [1] * len(prompt))

    current_input = torch.tensor(
        padded_inputs,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.tensor(
        masks,
        dtype=torch.long,
        device=device,
    )
    past_key_values = None
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    outputs: list[list[int]] = [[] for _ in range(batch_size)]
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
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        next_tokens = sample_batch(
            logits[:, -1],
            params,
            generators,
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_seconds.append(time.perf_counter() - started)

        for index, token in enumerate(next_tokens.tolist()):
            if finished[index]:
                continue
            outputs[index].append(token)
            if token in eos_token_ids:
                finished[index] = True

        if bool(finished.all()):
            break

        # 静态 batch 的形状不会缩小。已结束请求继续填 pad，占用计算槽位；
        # 活跃请求输入刚生成的 token。下一步会让调度器回收这些空槽。
        current_input = torch.where(
            finished,
            torch.full_like(next_tokens, pad_token_id),
            next_tokens,
        )[:, None]
        attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones(
                    (batch_size, 1),
                    dtype=attention_mask.dtype,
                    device=device,
                ),
            ),
            dim=1,
        )

    peak_memory_mib = 0.0
    if device.type == "cuda":
        peak_memory_mib = torch.cuda.max_memory_allocated(device) / 1024**2

    return BatchGenerationResult(
        output_token_ids=outputs,
        step_seconds=step_seconds,
        first_token_seconds=step_seconds[0],
        peak_memory_mib=peak_memory_mib,
    )

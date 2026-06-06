from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SamplingParams:
    """控制如何从模型输出的 logits 中选择下一个 token。"""

    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature 不能为负数")
        if self.top_k < 0:
            raise ValueError("top_k 不能为负数")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p 必须位于 (0, 1] 区间")

    @property
    def is_greedy(self) -> bool:
        # temperature=0 是一个明确约定：不做除法，直接选择最大 logits。
        return self.temperature == 0


def create_generator(
    params: SamplingParams,
    device: torch.device,
) -> torch.Generator | None:
    if params.is_greedy:
        return None
    generator = torch.Generator(device=device)
    if params.seed is not None:
        generator.manual_seed(params.seed)
    else:
        generator.seed()
    return generator


def filter_logits(
    logits: torch.Tensor,
    params: SamplingParams,
) -> torch.Tensor:
    """应用 temperature、top-k 和 top-p，返回过滤后的 FP32 logits。"""

    if params.is_greedy:
        return logits.float()

    filtered = logits.float() / params.temperature
    vocab_size = filtered.shape[-1]

    if 0 < params.top_k < vocab_size:
        # 直接按索引保留恰好 k 个 token，避免边界值相同时意外留下超过 k 个候选。
        top_indices = torch.topk(filtered, params.top_k, dim=-1).indices
        remove_mask = torch.ones_like(filtered, dtype=torch.bool)
        remove_mask.scatter_(dim=-1, index=top_indices, value=False)
        filtered = filtered.masked_fill(remove_mask, float("-inf"))

    if params.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            filtered,
            descending=True,
            dim=-1,
        )
        sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative_probabilities = sorted_probabilities.cumsum(dim=-1)

        # 当前 token 应在“加入它之前”的累计概率已经达到 top_p 时才被删除。
        # 这样概率最高的 token 永远会保留，候选集合也会刚好越过 top_p。
        remove_sorted = (
            cumulative_probabilities - sorted_probabilities
        ) >= params.top_p
        remove_mask = torch.zeros_like(remove_sorted).scatter(
            dim=-1,
            index=sorted_indices,
            src=remove_sorted,
        )
        filtered = filtered.masked_fill(remove_mask, float("-inf"))

    return filtered


def sample_next_token(
    logits: torch.Tensor,
    params: SamplingParams,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """从 `[batch, vocab]` logits 中为每条序列选择一个 token。"""

    if logits.ndim != 2:
        raise ValueError("logits 的形状必须是 [batch, vocab]")
    if params.is_greedy:
        return logits.argmax(dim=-1)

    filtered = filter_logits(logits, params)
    probabilities = torch.softmax(filtered, dim=-1)
    return torch.multinomial(
        probabilities,
        num_samples=1,
        generator=generator,
    ).squeeze(-1)


def create_generators(
    params: SamplingParams,
    device: torch.device,
    batch_size: int,
) -> list[torch.Generator | None]:
    """为 batch 中每条请求创建独立随机流。

    第 i 条请求使用 seed+i。这样第一条请求单独运行或放进 batch 时仍会得到相同结果，
    其他请求也不会因为前一条请求多抽了一次随机数而改变后续采样。
    """

    if params.is_greedy:
        return [None] * batch_size
    if params.seed is None:
        return [create_generator(params, device) for _ in range(batch_size)]

    generators = []
    for index in range(batch_size):
        generator = torch.Generator(device=device)
        generator.manual_seed(params.seed + index)
        generators.append(generator)
    return generators


def sample_batch(
    logits: torch.Tensor,
    params: SamplingParams,
    generators: list[torch.Generator | None],
) -> torch.Tensor:
    if logits.shape[0] != len(generators):
        raise ValueError("generator 数量必须和 batch size 一致")
    if params.is_greedy:
        return sample_next_token(logits, params)

    # torch.multinomial 只接收一个 generator。逐行采样让每条请求拥有独立随机流，
    # 这是请求可复现性比少量 Python 循环更重要的地方。
    return torch.stack(
        [
            sample_next_token(
                logits[index : index + 1],
                params,
                generator=generators[index],
            )[0]
            for index in range(logits.shape[0])
        ]
    )

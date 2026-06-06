from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

from toyvllm.config import ModelConfig
from toyvllm.layers.rotary_embedding import RotaryEmbedding
from toyvllm.models.qwen3 import Qwen3ForCausalLM


@dataclass(frozen=True)
class LoadedModel:
    model: Qwen3ForCausalLM
    load_seconds: float


def load_model(
    config: ModelConfig,
    *,
    device: str | torch.device = "cuda",
) -> LoadedModel:
    """用 meta 初始化和 safetensors 分片加载完整模型。"""

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但 torch.cuda.is_available() 为 False")

    started = time.perf_counter()

    # meta 参数只有形状和 dtype，没有真实内存。否则普通构造会先生成约 8 GB 的
    # FP32 随机参数，随后加载 BF16 权重时还要再占约 4 GB，8 GB 显卡无法承受。
    with torch.device("meta"):  # 只有 meta 没有具体随机化数据
        model = Qwen3ForCausalLM(config)

    index_path = config.model_path / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as file:
        weight_index = json.load(file)
    weight_map: dict[str, str] = weight_index["weight_map"]

    model_keys = set(model.state_dict().keys())
    weight_keys = set(weight_map)
    missing_weights = model_keys - weight_keys
    unexpected_weights = weight_keys - model_keys
    if missing_weights or unexpected_weights:
        raise ValueError(
            "模型结构与权重索引不一致："
            f"缺少 {sorted(missing_weights)[:5]}，"
            f"多出 {sorted(unexpected_weights)[:5]}"
        )

    loaded_keys: set[str] = set()
    shard_names = sorted(set(weight_map.values()))
    for shard_name in shard_names:
        shard_path = config.model_path / shard_name
        shard = load_file(str(shard_path), device=str(target_device))
        expected_in_shard = {
            name for name, filename in weight_map.items() if filename == shard_name
        }
        actual_in_shard = set(shard)
        if actual_in_shard != expected_in_shard:
            raise ValueError(f"权重分片内容与索引不一致：{shard_name}")

        # assign=True 让 Parameter 直接引用 safetensors 已经加载好的 CUDA 存储，
        # 避免再复制一次完整权重。
        incompatible = model.load_state_dict(shard, strict=False, assign=True)
        if incompatible.unexpected_keys:
            raise ValueError(f"加载到未知权重：{incompatible.unexpected_keys[:5]}")
        loaded_keys.update(actual_in_shard)
        del shard
        gc.collect()

    if loaded_keys != model_keys:
        raise ValueError("并非所有模型参数都完成加载")

    # RoPE inv_freq 是可推导的 buffer，不保存在 safetensors 中。meta 构造后必须
    # 在目标设备重建，否则第一次前向时会混用 meta 和 CUDA 张量。
    for module in model.modules():
        if isinstance(module, RotaryEmbedding):
            module.materialize(target_device)

    for name, parameter in model.named_parameters():
        if parameter.is_meta:
            raise RuntimeError(f"参数仍停留在 meta device：{name}")

    model.eval()
    return LoadedModel(
        model=model,
        load_seconds=time.perf_counter() - started,
    )


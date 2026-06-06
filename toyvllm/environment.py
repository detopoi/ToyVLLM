from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from toyvllm.config import ModelConfig


@dataclass(frozen=True)
class EnvironmentReport:
    python: str
    executable: str
    platform: str
    torch: str
    torch_cuda: str | None
    cuda_available: bool
    gpu_name: str | None
    gpu_memory_mib: int | None
    bf16_supported: bool
    model_path: str
    model_config_valid: bool

    def format(self) -> str:
        rows = [
            ("Python", self.python),
            ("解释器", self.executable),
            ("系统", self.platform),
            ("PyTorch", self.torch),
            ("PyTorch CUDA", str(self.torch_cuda)),
            ("CUDA 可用", str(self.cuda_available)),
            ("GPU", str(self.gpu_name)),
            ("GPU 显存", f"{self.gpu_memory_mib} MiB"),
            ("支持 BF16", str(self.bf16_supported)),
            ("模型目录", self.model_path),
            ("模型配置有效", str(self.model_config_valid)),
        ]
        width = max(len(name) for name, _ in rows)
        return "\n".join(f"{name:<{width}} : {value}" for name, value in rows)


def inspect_environment(model_path: str | Path) -> EnvironmentReport:
    config = ModelConfig.from_pretrained(model_path)
    cuda_available = torch.cuda.is_available()

    gpu_name = None
    gpu_memory_mib = None
    bf16_supported = False
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        gpu_name = properties.name
        gpu_memory_mib = round(properties.total_memory / 1024**2)
        bf16_supported = torch.cuda.is_bf16_supported()

    return EnvironmentReport(
        python=platform.python_version(),
        executable=sys.executable,
        platform=platform.platform(),
        torch=torch.__version__,
        torch_cuda=torch.version.cuda,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        gpu_memory_mib=gpu_memory_mib,
        bf16_supported=bf16_supported,
        model_path=str(config.model_path),
        model_config_valid=True,
    )


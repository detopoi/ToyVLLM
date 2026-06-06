from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DTYPE_BYTES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
}


@dataclass(frozen=True)
class ModelConfig:
    """推理时真正需要关心的 Qwen3 结构参数。

    Hugging Face 的 config.json 包含很多训练和兼容性字段。这里主动挑出推理
    需要的字段，是为了让后续每个张量维度都能追溯到一个明确配置，而不是在
    模型代码里散落魔法数字。
    """

    model_path: Path
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    torch_dtype: str
    tie_word_embeddings: bool
    bos_token_id: int
    eos_token_id: int
    generation_eos_token_ids: tuple[int, ...] = ()
    generation_do_sample: bool = True
    generation_temperature: float = 0.6
    generation_top_k: int = 20
    generation_top_p: float = 0.95

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "ModelConfig":
        path = Path(model_path).expanduser().resolve()
        config_path = path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"找不到模型配置文件：{config_path}")

        with config_path.open("r", encoding="utf-8") as file:
            raw: dict[str, Any] = json.load(file)
        generation_config_path = path / "generation_config.json"
        generation_raw: dict[str, Any] = {}
        if generation_config_path.is_file():
            with generation_config_path.open("r", encoding="utf-8") as file:
                generation_raw = json.load(file)

        if raw.get("model_type") != "qwen3":
            raise ValueError(
                f"当前教学项目只支持 qwen3，实际得到：{raw.get('model_type')!r}"
            )

        config = cls(
            model_path=path,
            hidden_size=int(raw["hidden_size"]),
            intermediate_size=int(raw["intermediate_size"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            num_key_value_heads=int(raw["num_key_value_heads"]),
            head_dim=int(raw["head_dim"]),
            vocab_size=int(raw["vocab_size"]),
            max_position_embeddings=int(raw["max_position_embeddings"]),
            rms_norm_eps=float(raw["rms_norm_eps"]),
            rope_theta=float(raw["rope_theta"]),
            torch_dtype=str(raw["torch_dtype"]),
            tie_word_embeddings=bool(raw["tie_word_embeddings"]),
            bos_token_id=int(raw["bos_token_id"]),
            eos_token_id=int(raw["eos_token_id"]),
            generation_eos_token_ids=cls._normalize_token_ids(
                generation_raw.get("eos_token_id", raw["eos_token_id"])
            ),
            generation_do_sample=bool(generation_raw.get("do_sample", False)),
            generation_temperature=float(
                generation_raw.get("temperature", 1.0)
            ),
            generation_top_k=int(generation_raw.get("top_k", 0)),
            generation_top_p=float(generation_raw.get("top_p", 1.0)),
        )
        config._validate()
        return config

    @staticmethod
    def _normalize_token_ids(value: int | list[int]) -> tuple[int, ...]:
        if isinstance(value, int):
            return (value,)
        return tuple(int(token_id) for token_id in value)

    def _validate(self) -> None:
        # 每个 Query Head 占 head_dim 维，所以拼接所有头后必须正好回到 hidden_size。
        if self.num_attention_heads * self.head_dim != self.hidden_size:
            raise ValueError("num_attention_heads * head_dim 必须等于 hidden_size")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("Query Head 数必须能被 KV Head 数整除")
        if self.torch_dtype not in DTYPE_BYTES:
            raise ValueError(f"暂不支持权重类型：{self.torch_dtype}")

    @property
    def queries_per_kv(self) -> int:
        """GQA 中，每一组 Key/Value 被多少个 Query Head 共享。"""

        return self.num_attention_heads // self.num_key_value_heads

    @property
    def kv_cache_bytes_per_token(self) -> int:
        """整模型为一个 token 保存 KV Cache 所需的字节数。

        每层都要同时保存 K 和 V，因此公式中有一个 2。GQA 只缓存 KV Head，
        不按 Query Head 数缓存，这正是 GQA 能节省显存的重要原因。
        """

        return (
            self.num_hidden_layers
            * 2
            * self.num_key_value_heads
            * self.head_dim
            * DTYPE_BYTES[self.torch_dtype]
        )

    def kv_cache_mib(self, num_tokens: int) -> float:
        if num_tokens < 0:
            raise ValueError("num_tokens 不能为负数")
        return self.kv_cache_bytes_per_token * num_tokens / 1024**2

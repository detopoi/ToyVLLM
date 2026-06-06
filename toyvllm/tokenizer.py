from __future__ import annotations

from pathlib import Path
from typing import Sequence, TypedDict

from transformers import AutoTokenizer, PreTrainedTokenizerBase


class ChatMessage(TypedDict):
    role: str
    content: str


class Tokenizer:
    """Qwen3 tokenizer 的薄封装。

    这里保留 Hugging Face tokenizer，是因为分词算法不是本项目要重复实现的重点。
    我们只把“普通文本”和“聊天消息”这两个入口固定下来，后续引擎只接收 token id，
    从而让文本处理和 GPU 模型计算保持清晰边界。
    """

    def __init__(self, model_path: str | Path) -> None:
        path = Path(model_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"找不到模型目录：{path}")

        self.model_path = path
        self.backend: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            path,
            local_files_only=True,
            use_fast=True,
        )

    @property
    def eos_token_id(self) -> int:
        token_id = self.backend.eos_token_id
        if token_id is None:
            raise ValueError("tokenizer 没有配置 eos_token_id")
        return token_id

    @property
    def pad_token_id(self) -> int:
        token_id = self.backend.pad_token_id
        if token_id is None:
            raise ValueError("tokenizer 没有配置 pad_token_id")
        return token_id

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return self.backend.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = False) -> str:
        return self.backend.decode(
            list(token_ids),
            skip_special_tokens=skip_special_tokens,
        )

    def render_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str:
        self._validate_messages(messages)
        return self.backend.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )

    def encode_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> list[int]:
        self._validate_messages(messages)
        token_ids = self.backend.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
        return list(token_ids)

    @staticmethod
    def _validate_messages(messages: Sequence[ChatMessage]) -> None:
        if not messages:
            raise ValueError("聊天消息不能为空")
        allowed_roles = {"system", "user", "assistant"}
        for index, message in enumerate(messages):
            if message["role"] not in allowed_roles:
                raise ValueError(f"第 {index} 条消息的 role 不合法：{message['role']}")
            if not isinstance(message["content"], str):
                raise TypeError(f"第 {index} 条消息的 content 必须是字符串")


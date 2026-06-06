import unittest

from toyvllm.tokenizer import Tokenizer


class TokenizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = Tokenizer("Qwen3-1.7B")

    def test_plain_text_round_trip(self) -> None:
        text = "你好，推理引擎！"
        token_ids = self.tokenizer.encode(text)
        self.assertEqual(self.tokenizer.decode(token_ids), text)

    def test_chat_template_adds_assistant_prompt(self) -> None:
        messages = [{"role": "user", "content": "你好"}]
        rendered = self.tokenizer.render_chat(messages, enable_thinking=False)
        self.assertIn("<|im_start|>user", rendered)
        self.assertIn("<|im_start|>assistant", rendered)
        self.assertIn("<think>", rendered)

    def test_empty_chat_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.tokenizer.encode_chat([])


if __name__ == "__main__":
    unittest.main()


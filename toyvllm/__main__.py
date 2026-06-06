from __future__ import annotations

import argparse

from toyvllm.config import ModelConfig
from toyvllm.environment import inspect_environment
from toyvllm.generation import generate_greedy_cached, generate_greedy_naive
from toyvllm.sampling import SamplingParams
from toyvllm.tokenizer import ChatMessage, Tokenizer
from toyvllm.weight_loader import load_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Toy vLLM 教学项目")
    parser.add_argument("--model", default="Qwen3-1.7B", help="本地模型目录")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("env", help="检查 Python、CUDA、GPU 和模型配置")

    tokenize_parser = subparsers.add_parser("tokenize", help="演示聊天模板和分词")
    tokenize_parser.add_argument("prompt", nargs="?", default="用一句话解释 KV Cache。")
    tokenize_parser.add_argument(
        "--thinking",
        action="store_true",
        help="启用 Qwen3 thinking 模式",
    )

    generate_parser = subparsers.add_parser("generate", help="运行文本生成")
    generate_parser.add_argument("prompt", nargs="?", default="用一句话解释 KV Cache。")
    generate_parser.add_argument("--max-new-tokens", type=int, default=16)
    generate_parser.add_argument("--thinking", action="store_true")
    generate_parser.add_argument(
        "--backend",
        choices=("naive", "cached"),
        default="cached",
        help="naive 每步重算完整序列；cached 复用 KV Cache",
    )
    generate_parser.add_argument(
        "--sample",
        action="store_true",
        help="使用模型 generation_config.json 中的采样参数",
    )
    generate_parser.add_argument("--temperature", type=float, default=None)
    generate_parser.add_argument("--top-k", type=int, default=None)
    generate_parser.add_argument("--top-p", type=float, default=None)
    generate_parser.add_argument("--seed", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "env":
        report = inspect_environment(args.model)
        config = ModelConfig.from_pretrained(args.model)
        print(report.format())
        print(f"KV Cache/token : {config.kv_cache_bytes_per_token / 1024:.1f} KiB")
        print(f"KV Cache/1024  : {config.kv_cache_mib(1024):.1f} MiB")
        return

    if args.command == "tokenize":
        tokenizer = Tokenizer(args.model)
        messages: list[ChatMessage] = [{"role": "user", "content": args.prompt}]
        rendered = tokenizer.render_chat(messages, enable_thinking=args.thinking)
        token_ids = tokenizer.encode_chat(messages, enable_thinking=args.thinking)
        print("聊天模板展开结果：")
        print(rendered)
        print(f"\nToken 数：{len(token_ids)}")
        print(f"Token IDs：{token_ids}")
        print(f"解码结果：\n{tokenizer.decode(token_ids)}")
        return

    if args.command == "generate":
        config = ModelConfig.from_pretrained(args.model)
        tokenizer = Tokenizer(args.model)
        messages: list[ChatMessage] = [{"role": "user", "content": args.prompt}]
        prompt_token_ids = tokenizer.encode_chat(
            messages,
            enable_thinking=args.thinking,
        )

        loaded = load_model(config, device="cuda")
        sampling_requested = args.sample or any(
            value is not None
            for value in (args.temperature, args.top_k, args.top_p, args.seed)
        )
        if sampling_requested:
            sampling_params = SamplingParams(
                temperature=(
                    config.generation_temperature
                    if args.temperature is None
                    else args.temperature
                ),
                top_k=(
                    config.generation_top_k
                    if args.top_k is None
                    else args.top_k
                ),
                top_p=(
                    config.generation_top_p
                    if args.top_p is None
                    else args.top_p
                ),
                seed=args.seed,
            )
        else:
            sampling_params = SamplingParams()

        generate = (
            generate_greedy_cached
            if args.backend == "cached"
            else generate_greedy_naive
        )
        result = generate(
            loaded.model,
            prompt_token_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=set(config.generation_eos_token_ids),
            sampling_params=sampling_params,
        )
        print(f"模型加载：{loaded.load_seconds:.2f} s")
        print(f"推理后端：{args.backend}")
        if sampling_params.is_greedy:
            print("解码策略：greedy")
        else:
            print(
                "解码策略：sampling "
                f"(temperature={sampling_params.temperature}, "
                f"top_k={sampling_params.top_k}, "
                f"top_p={sampling_params.top_p}, "
                f"seed={sampling_params.seed})"
            )
        print(f"输入 token：{len(prompt_token_ids)}")
        print(f"输出 token：{len(result.output_token_ids)}")
        print(f"TTFT：{result.ttft_ms:.2f} ms")
        print(f"TPOT：{result.tpot_ms:.2f} ms")
        print(f"输出吞吐：{result.output_tokens_per_second:.2f} tokens/s")
        print(f"Decode 吞吐：{result.decode_tokens_per_second:.2f} tokens/s")
        print(f"峰值显存：{result.peak_memory_mib:.1f} MiB")
        print("\n模型输出：")
        print(tokenizer.decode(result.output_token_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()

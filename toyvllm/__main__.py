from __future__ import annotations

import argparse

from toyvllm.config import ModelConfig
from toyvllm.engine import ContinuousBatchEngine, PagedContinuousBatchEngine
from toyvllm.environment import inspect_environment
from toyvllm.generation import (
    generate_greedy_cached,
    generate_greedy_naive,
    generate_static_batch,
)
from toyvllm.sampling import SamplingParams
from toyvllm.tokenizer import ChatMessage, Tokenizer
from toyvllm.weight_loader import load_model


def add_sampling_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sample",
        action="store_true",
        help="使用模型 generation_config.json 中的采样参数",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="温度；0 表示 greedy，正数启用采样",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="只保留概率最高的 k 个候选；0 表示不限制",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="nucleus sampling 累计概率阈值，范围 (0, 1]",
    )
    parser.add_argument("--seed", type=int, default=None, help="采样随机种子")


def resolve_sampling_params(
    args: argparse.Namespace,
    config: ModelConfig,
) -> SamplingParams:
    sampling_requested = args.sample or any(
        value is not None
        for value in (args.temperature, args.top_k, args.top_p, args.seed)
    )
    if not sampling_requested:
        return SamplingParams()
    return SamplingParams(
        temperature=(
            config.generation_temperature
            if args.temperature is None
            else args.temperature
        ),
        top_k=config.generation_top_k if args.top_k is None else args.top_k,
        top_p=config.generation_top_p if args.top_p is None else args.top_p,
        seed=args.seed,
    )


def format_sampling_params(params: SamplingParams) -> str:
    if params.is_greedy:
        return "greedy"
    return (
        "sampling "
        f"(temperature={params.temperature}, "
        f"top_k={params.top_k}, "
        f"top_p={params.top_p}, "
        f"seed={params.seed})"
    )


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
    add_sampling_arguments(generate_parser)

    batch_parser = subparsers.add_parser("batch", help="运行静态批处理生成")
    batch_parser.add_argument(
        "prompts",
        nargs="+",
        help="一条或多条 prompt，不同长度会自动左 padding",
    )
    batch_parser.add_argument("--max-new-tokens", type=int, default=16)
    batch_parser.add_argument("--thinking", action="store_true")
    add_sampling_arguments(batch_parser)

    continuous_parser = subparsers.add_parser(
        "continuous",
        help="运行带 FIFO Scheduler 的连续批处理",
    )
    continuous_parser.add_argument(
        "prompts",
        nargs="+",
        help="提交到 waiting 队列的一条或多条 prompt",
    )
    continuous_parser.add_argument("--max-new-tokens", type=int, default=16)
    continuous_parser.add_argument("--max-num-seqs", type=int, default=4)
    continuous_parser.add_argument(
        "--cache-backend",
        choices=("continuous", "paged"),
        default="continuous",
    )
    continuous_parser.add_argument("--num-kv-blocks", type=int, default=64)
    continuous_parser.add_argument("--block-size", type=int, default=16)
    continuous_parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help=(
            "每轮 Decode + Prefill 的 token budget；设置后为 paged 后端启用 "
            "Chunked Prefill"
        ),
    )
    continuous_parser.add_argument(
        "--max-prefill-chunk-size",
        type=int,
        default=256,
        help="Chunked Prefill 中单条请求每轮最多处理的 Prompt token 数",
    )
    continuous_parser.add_argument(
        "--max-mixed-prefill-tokens",
        type=int,
        default=None,
        help="已有 Decode 时，同轮 Prefill token 总上限；默认仍使用剩余总预算",
    )
    continuous_parser.add_argument(
        "--paged-attention",
        choices=(
            "auto",
            "gather",
            "paged",
            "triton",
            "triton-fixed",
            "triton-grouped",
        ),
        default="auto",
        help="paged 后端的 Attention：9B gather、9C PyTorch 或 Triton Kernel",
    )
    continuous_parser.add_argument("--thinking", action="store_true")
    continuous_parser.add_argument(
        "--show-schedule",
        action="store_true",
        help="打印每轮 decode/prefill 的 request id",
    )
    add_sampling_arguments(continuous_parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "env":
        report = inspect_environment(args.model)
        config = ModelConfig.from_pretrained(args.model)
        print(report.format())
        print(f"KV Cache/token : {config.kv_cache_bytes_per_token / 1024:.1f} KiB")
        print(f"KV Cache/1024  : {config.kv_cache_mib(1024):.1f} MiB")
        print(f"KV Block/16    : {config.kv_cache_block_mib(16):.2f} MiB")
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
        sampling_params = resolve_sampling_params(args, config)

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
        print(f"解码策略：{format_sampling_params(sampling_params)}")
        print(f"输入 token：{len(prompt_token_ids)}")
        print(f"输出 token：{len(result.output_token_ids)}")
        print(f"TTFT：{result.ttft_ms:.2f} ms")
        print(f"TPOT：{result.tpot_ms:.2f} ms")
        print(f"输出吞吐：{result.output_tokens_per_second:.2f} tokens/s")
        print(f"Decode 吞吐：{result.decode_tokens_per_second:.2f} tokens/s")
        print(f"峰值显存：{result.peak_memory_mib:.1f} MiB")
        print("\n模型输出：")
        print(tokenizer.decode(result.output_token_ids, skip_special_tokens=True))
        return

    if args.command == "batch":
        config = ModelConfig.from_pretrained(args.model)
        tokenizer = Tokenizer(args.model)
        prompt_token_ids = [
            tokenizer.encode_chat(
                [{"role": "user", "content": prompt}],
                enable_thinking=args.thinking,
            )
            for prompt in args.prompts
        ]
        sampling_params = resolve_sampling_params(args, config)
        loaded = load_model(config, device="cuda")
        result = generate_static_batch(
            loaded.model,
            prompt_token_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=set(config.generation_eos_token_ids),
            pad_token_id=tokenizer.pad_token_id,
            sampling_params=sampling_params,
        )

        print(f"模型加载：{loaded.load_seconds:.2f} s")
        print(f"Batch size：{len(args.prompts)}")
        print(f"输入长度：{[len(ids) for ids in prompt_token_ids]}")
        print(f"解码策略：{format_sampling_params(sampling_params)}")
        print(f"Batch TTFT：{result.ttft_ms:.2f} ms")
        print(f"每轮 TPOT：{result.tpot_ms:.2f} ms")
        print(f"总输出吞吐：{result.output_tokens_per_second:.2f} tokens/s")
        print(f"峰值显存：{result.peak_memory_mib:.1f} MiB")
        for index, token_ids in enumerate(result.output_token_ids):
            print(f"\n[{index}] {args.prompts[index]}")
            print(tokenizer.decode(token_ids, skip_special_tokens=True))
        return

    if args.command == "continuous":
        config = ModelConfig.from_pretrained(args.model)
        tokenizer = Tokenizer(args.model)
        sampling_params = resolve_sampling_params(args, config)
        loaded = load_model(config, device="cuda")
        if args.cache_backend == "paged":
            engine = PagedContinuousBatchEngine(
                loaded.model,
                max_num_seqs=args.max_num_seqs,
                pad_token_id=tokenizer.pad_token_id,
                num_blocks=args.num_kv_blocks,
                block_size=args.block_size,
                attention_backend=args.paged_attention,
                max_num_batched_tokens=args.max_num_batched_tokens,
                max_prefill_chunk_size=args.max_prefill_chunk_size,
                max_mixed_prefill_tokens=args.max_mixed_prefill_tokens,
            )
        else:
            engine = ContinuousBatchEngine(
                loaded.model,
                max_num_seqs=args.max_num_seqs,
                pad_token_id=tokenizer.pad_token_id,
            )
        for prompt in args.prompts:
            prompt_ids = tokenizer.encode_chat(
                [{"role": "user", "content": prompt}],
                enable_thinking=args.thinking,
            )
            engine.add_request(
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                eos_token_ids=set(config.generation_eos_token_ids),
                sampling_params=sampling_params,
            )
        result = engine.run()

        print(f"模型加载：{loaded.load_seconds:.2f} s")
        print(f"请求数：{len(args.prompts)}")
        print(f"最大并发：{args.max_num_seqs}")
        print(f"缓存后端：{args.cache_backend}")
        if args.cache_backend == "paged":
            print(
                f"KV Blocks：{args.num_kv_blocks} × "
                f"{args.block_size} tokens"
            )
            print(f"Paged Attention：{engine.attention_backend}")
            if engine.chunked_prefill_enabled:
                print(
                    "Chunked Prefill："
                    f"budget={engine.max_num_batched_tokens}, "
                    f"max_chunk={engine.max_prefill_chunk_size}, "
                    "mixed_prefill="
                    f"{engine.max_mixed_prefill_tokens or 'unbounded'}"
                )
        print(f"解码策略：{format_sampling_params(sampling_params)}")
        print(f"调度轮数：{len(result.iterations)}")
        print(f"总输出吞吐：{result.output_tokens_per_second:.2f} tokens/s")
        print(f"请求吞吐：{result.requests_per_second:.2f} requests/s")
        print(f"完成顺序：{list(result.completion_order)}")
        print(f"峰值显存：{result.peak_memory_mib:.1f} MiB")
        if args.cache_backend == "paged":
            print(
                "结束后空闲 Blocks："
                f"{engine.block_manager.stats.num_free_blocks}/"
                f"{engine.block_manager.stats.num_total_blocks}"
            )

        if args.show_schedule:
            print("\n调度轨迹：")
            for iteration in result.iterations:
                print(
                    f"step={iteration.step:02d} "
                    f"decode={list(iteration.decode_request_ids)} "
                    "prefill="
                    f"{list(zip(iteration.prefill_request_ids, iteration.prefill_token_counts))}"
                )

        for sequence in result.sequences:
            print(f"\n[{sequence.request_id}] {args.prompts[sequence.request_id]}")
            print(
                tokenizer.decode(
                    sequence.output_token_ids,
                    skip_special_tokens=True,
                )
            )


if __name__ == "__main__":
    main()

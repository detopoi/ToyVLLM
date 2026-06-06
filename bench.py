from __future__ import annotations

import argparse
import statistics

from toyvllm.benchmark import append_record, append_result, benchmark
from toyvllm.config import ModelConfig
from toyvllm.generation import GenerationResult, generate_greedy_naive
from toyvllm.tokenizer import ChatMessage, Tokenizer
from toyvllm.weight_loader import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy vLLM 性能基准")
    parser.add_argument("--model", default="Qwen3-1.7B")
    parser.add_argument("--backend", choices=("tokenizer", "naive"), default="tokenizer")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--prompt", default="用一句话解释 KV Cache。")
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--save",
        default=None,
        help="可选的 JSONL 结果文件，例如 benchmarks/results.jsonl",
    )
    args = parser.parse_args()

    tokenizer = Tokenizer(args.model)
    if args.backend == "naive":
        run_naive_benchmark(args, tokenizer)
        return

    warmup = 10 if args.warmup is None else args.warmup
    iterations = 100 if args.iterations is None else args.iterations
    label = args.label or "stage-01-tokenizer"
    text = "大模型推理时，KV Cache 可以避免重复计算历史 token 的 Key 和 Value。"
    token_count = len(tokenizer.encode(text))

    result = benchmark(
        "Tokenizer encode（当前不是模型推理性能）",
        lambda: tokenizer.encode(text),
        warmup=warmup,
        iterations=iterations,
        items_per_iteration=token_count,
        unit="tokens",
    )
    print(result.format())
    if args.save:
        append_result(
            args.save,
            result,
            label=label,
            metadata={"model": args.model, "text_tokens": token_count},
        )
        print(f"\n结果已追加到：{args.save}")
    print("\n模型推理 benchmark：python bench.py --backend naive")


def run_naive_benchmark(args: argparse.Namespace, tokenizer: Tokenizer) -> None:
    warmup = 1 if args.warmup is None else args.warmup
    iterations = 3 if args.iterations is None else args.iterations
    label = args.label or "stage-04-naive"
    config = ModelConfig.from_pretrained(args.model)
    messages: list[ChatMessage] = [{"role": "user", "content": args.prompt}]
    prompt_token_ids = tokenizer.encode_chat(messages, enable_thinking=False)
    loaded = load_model(config, device="cuda")

    def generate() -> GenerationResult:
        return generate_greedy_naive(
            loaded.model,
            prompt_token_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=set(config.generation_eos_token_ids),
        )

    for _ in range(warmup):
        generate()
    results = [generate() for _ in range(iterations)]

    total_tokens = sum(len(result.output_token_ids) for result in results)
    total_seconds = sum(sum(result.step_seconds) for result in results)
    metrics = {
        "backend": "naive",
        "iterations": iterations,
        "mean_ttft_ms": statistics.fmean(result.ttft_ms for result in results),
        "mean_tpot_ms": statistics.fmean(result.tpot_ms for result in results),
        "output_tokens_per_second": total_tokens / total_seconds,
        "peak_memory_mib": max(result.peak_memory_mib for result in results),
        "model_load_seconds": loaded.load_seconds,
    }
    print("Naive greedy generation")
    print(f"  模型加载 : {metrics['model_load_seconds']:.2f} s")
    print(f"  平均 TTFT: {metrics['mean_ttft_ms']:.2f} ms")
    print(f"  平均 TPOT: {metrics['mean_tpot_ms']:.2f} ms")
    print(f"  输出吞吐 : {metrics['output_tokens_per_second']:.2f} tokens/s")
    print(f"  峰值显存 : {metrics['peak_memory_mib']:.1f} MiB")

    if args.save:
        append_record(
            args.save,
            label=label,
            metrics=metrics,
            metadata={
                "model": args.model,
                "prompt": args.prompt,
                "prompt_tokens": len(prompt_token_ids),
                "max_new_tokens": args.max_new_tokens,
                "warmup": warmup,
                "decoding": "greedy",
            },
        )
        print(f"\n结果已追加到：{args.save}")


if __name__ == "__main__":
    main()

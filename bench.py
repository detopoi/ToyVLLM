from __future__ import annotations

import argparse
import statistics
import time

import torch

from toyvllm.benchmark import append_record, append_result, benchmark
from toyvllm.config import ModelConfig
from toyvllm.engine import (
    ContinuousBatchEngine,
    ContinuousBatchResult,
    PagedContinuousBatchEngine,
)
from toyvllm.generation import (
    BatchGenerationResult,
    GenerationResult,
    generate_greedy_cached,
    generate_greedy_naive,
    generate_static_batch,
)
from toyvllm.kernels.paged_attention import (
    clear_triton_autotune_cache,
    get_triton_autotune_results,
)
from toyvllm.sampling import SamplingParams
from toyvllm.tokenizer import ChatMessage, Tokenizer
from toyvllm.weight_loader import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy vLLM 性能基准")
    parser.add_argument("--model", default="Qwen3-1.7B")
    parser.add_argument(
        "--backend",
        choices=(
            "tokenizer",
            "naive",
            "cached",
            "static",
            "continuous",
            "paged",
        ),
        default="tokenizer",
    )
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--prompt", default="用一句话解释 KV Cache。")
    parser.add_argument(
        "--prompt-repeat",
        type=int,
        default=1,
        help="重复用户 prompt，用于观察长上下文下的性能变化",
    )
    parser.add_argument("--label", default=None)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--short-new-tokens", type=int, default=4)
    parser.add_argument("--num-kv-blocks", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--save",
        default=None,
        help="可选的 JSONL 结果文件，例如 benchmarks/results.jsonl",
    )
    args = parser.parse_args()

    tokenizer = Tokenizer(args.model)
    if args.backend == "paged":
        run_paged_benchmark(args, tokenizer)
        return
    if args.backend == "continuous":
        run_scheduler_benchmark(args, tokenizer)
        return
    if args.backend == "static":
        run_static_batch_benchmark(args, tokenizer)
        return
    if args.backend in {"naive", "cached"}:
        run_generation_benchmark(args, tokenizer)
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
    print("\n模型推理 benchmark：python bench.py --backend cached")


def resolve_sampling_params(
    args: argparse.Namespace,
    config: ModelConfig,
) -> SamplingParams:
    requested = args.sample or any(
        value is not None
        for value in (args.temperature, args.top_k, args.top_p, args.seed)
    )
    if not requested:
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


def run_paged_benchmark(
    args: argparse.Namespace,
    tokenizer: Tokenizer,
) -> None:
    """比较 10C 的常驻 BlockTable Workspace 与 Triton autotune。"""

    if args.batch_size <= 0 or args.num_requests <= 0:
        raise ValueError("--batch-size 和 --num-requests 必须大于 0")
    warmup = 1 if args.warmup is None else args.warmup
    iterations = 3 if args.iterations is None else args.iterations
    label = args.label or "stage-10c-resident-metadata-autotune"
    config = ModelConfig.from_pretrained(args.model)
    prompt_ids = tokenizer.encode_chat(
        [{"role": "user", "content": args.prompt}],
        enable_thinking=False,
    )
    prompts = [prompt_ids] * args.num_requests
    limits = [
        args.short_new_tokens if index % 2 == 0 else args.max_new_tokens
        for index in range(args.num_requests)
    ]
    loaded = load_model(config, device="cuda")
    sampling_params = resolve_sampling_params(args, config)

    def run_engine(
        attention_backend: str,
        *,
        resident_block_tables: bool = True,
    ) -> ContinuousBatchResult:
        engine = PagedContinuousBatchEngine(
            loaded.model,
            max_num_seqs=args.batch_size,
            pad_token_id=tokenizer.pad_token_id,
            num_blocks=args.num_kv_blocks,
            block_size=args.block_size,
            attention_backend=attention_backend,
            resident_block_tables=resident_block_tables,
        )
        for prompt, limit in zip(prompts, limits):
            engine.add_request(
                prompt,
                max_new_tokens=limit,
                eos_token_ids=set(config.generation_eos_token_ids),
                sampling_params=sampling_params,
            )
        return engine.run()

    variants = {
        "gather": ("gather", True),
        "transient_fixed": ("triton-fixed", False),
        "resident_fixed": ("triton-fixed", True),
        "resident_autotune": ("triton", True),
    }
    clear_triton_autotune_cache()
    for _ in range(warmup):
        for backend, resident in variants.values():
            run_engine(
                backend,
                resident_block_tables=resident,
            )

    results: dict[str, list[ContinuousBatchResult]] = {
        name: [] for name in variants
    }
    forward_order = tuple(variants)
    reverse_order = tuple(reversed(forward_order))
    for iteration in range(iterations):
        order = forward_order if iteration % 2 == 0 else reverse_order
        for name in order:
            backend, resident = variants[name]
            results[name].append(
                run_engine(
                    backend,
                    resident_block_tables=resident,
                )
            )

    def tokens_per_second(name: str) -> float:
        total_tokens = sum(
            result.total_output_tokens for result in results[name]
        )
        total_seconds = sum(result.total_seconds for result in results[name])
        return total_tokens / total_seconds

    def peak_memory(name: str) -> float:
        return max(result.peak_memory_mib for result in results[name])

    gather_tps = tokens_per_second("gather")
    transient_tps = tokens_per_second("transient_fixed")
    resident_fixed_tps = tokens_per_second("resident_fixed")
    autotuned_tps = tokens_per_second("resident_autotune")
    autotune_results = get_triton_autotune_results()
    selected_num_warps = sorted(
        {
            int(result["num_warps"])
            for result in autotune_results.values()
        }
    )
    fastest_num_warps = sorted(
        {
            int(result["fastest_num_warps"])
            for result in autotune_results.values()
        }
    )
    metrics = {
        "backend": "paged",
        "max_num_seqs": args.batch_size,
        "num_requests": args.num_requests,
        "iterations": iterations,
        "gather_tokens_per_second": gather_tps,
        "transient_fixed_tokens_per_second": transient_tps,
        "resident_fixed_tokens_per_second": resident_fixed_tps,
        "autotuned_tokens_per_second": autotuned_tps,
        "resident_metadata_speedup": resident_fixed_tps / transient_tps,
        "autotune_speedup": autotuned_tps / resident_fixed_tps,
        "speedup_vs_gather": autotuned_tps / gather_tps,
        "gather_peak_memory_mib": peak_memory("gather"),
        "transient_fixed_peak_memory_mib": peak_memory("transient_fixed"),
        "resident_fixed_peak_memory_mib": peak_memory("resident_fixed"),
        "autotuned_peak_memory_mib": peak_memory("resident_autotune"),
        "autotune_config_count": len(autotune_results),
        "selected_num_warps": selected_num_warps,
        "fastest_num_warps": fastest_num_warps,
        "num_kv_blocks": args.num_kv_blocks,
        "block_size": args.block_size,
    }

    print("Paged Attention 10C: resident BlockTable + autotune")
    print(f"  请求数/最大并发 : {args.num_requests}/{args.batch_size}")
    print(f"  生成上限       : {limits}")
    print(
        f"  KV Blocks      : {args.num_kv_blocks} × "
        f"{args.block_size} tokens"
    )
    print(f"  Gather + SDPA         : {gather_tps:.2f} tokens/s")
    print(f"  瞬时元数据 + 4 warps : {transient_tps:.2f} tokens/s")
    print(f"  常驻元数据 + 4 warps : {resident_fixed_tps:.2f} tokens/s")
    print(f"  常驻元数据 + autotune: {autotuned_tps:.2f} tokens/s")
    print(
        "  常驻元数据倍率       : "
        f"{metrics['resident_metadata_speedup']:.2f}x"
    )
    print(
        "  Autotune 倍率        : "
        f"{metrics['autotune_speedup']:.2f}x"
    )
    print(
        "  Triton/Gather 倍率   : "
        f"{metrics['speedup_vs_gather']:.2f}x"
    )
    print(
        "  微基准最快 warps     : "
        f"{fastest_num_warps}"
    )
    print(
        "  保护后选中 warps     : "
        f"{selected_num_warps}（{len(autotune_results)} 种形状）"
    )
    print(
        "  Gather/瞬时/常驻/调优 峰值显存: "
        f"{metrics['gather_peak_memory_mib']:.1f} / "
        f"{metrics['transient_fixed_peak_memory_mib']:.1f} / "
        f"{metrics['resident_fixed_peak_memory_mib']:.1f} / "
        f"{metrics['autotuned_peak_memory_mib']:.1f} MiB"
    )

    if args.save:
        append_record(
            args.save,
            label=label,
            metrics=metrics,
            metadata={
                "model": args.model,
                "prompt_tokens": len(prompt_ids),
                "request_max_new_tokens": limits,
                "warmup": warmup,
                "decoding": (
                    "greedy" if sampling_params.is_greedy else "sampling"
                ),
            },
        )
        print(f"\n结果已追加到：{args.save}")


def run_scheduler_benchmark(
    args: argparse.Namespace,
    tokenizer: Tokenizer,
) -> None:
    if args.batch_size <= 0 or args.num_requests <= 0:
        raise ValueError("--batch-size 和 --num-requests 必须大于 0")
    if not 0 < args.short_new_tokens <= args.max_new_tokens:
        raise ValueError("--short-new-tokens 必须在 1 到 max-new-tokens 之间")

    warmup = 1 if args.warmup is None else args.warmup
    iterations = 3 if args.iterations is None else args.iterations
    label = args.label or "stage-08-continuous-batch"
    config = ModelConfig.from_pretrained(args.model)
    prompt_ids = tokenizer.encode_chat(
        [{"role": "user", "content": args.prompt}],
        enable_thinking=False,
    )
    prompts = [prompt_ids] * args.num_requests
    limits = [
        args.short_new_tokens if index % 2 == 0 else args.max_new_tokens
        for index in range(args.num_requests)
    ]
    loaded = load_model(config, device="cuda")
    sampling_params = resolve_sampling_params(args, config)

    def run_static_workload() -> dict[str, object]:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        total_tokens = 0
        completion_seconds: list[float] = []
        for offset in range(0, args.num_requests, args.batch_size):
            batch_prompts = prompts[offset : offset + args.batch_size]
            batch_limits = limits[offset : offset + args.batch_size]
            batch_started_at = time.perf_counter() - started
            result = generate_static_batch(
                loaded.model,
                batch_prompts,
                max_new_tokens=batch_limits,
                eos_token_ids=set(config.generation_eos_token_ids),
                pad_token_id=tokenizer.pad_token_id,
                sampling_params=sampling_params,
            )
            total_tokens += result.total_output_tokens
            completion_seconds.extend(
                batch_started_at + seconds
                for seconds in result.completion_seconds
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak = (
            torch.cuda.max_memory_allocated() / 1024**2
            if torch.cuda.is_available()
            else 0.0
        )
        return {
            "seconds": elapsed,
            "tokens": float(total_tokens),
            "peak_memory_mib": peak,
            "completion_seconds": completion_seconds,
        }

    def run_continuous_workload() -> ContinuousBatchResult:
        engine = ContinuousBatchEngine(
            loaded.model,
            max_num_seqs=args.batch_size,
            pad_token_id=tokenizer.pad_token_id,
        )
        for prompt, limit in zip(prompts, limits):
            engine.add_request(
                prompt,
                max_new_tokens=limit,
                eos_token_ids=set(config.generation_eos_token_ids),
                sampling_params=sampling_params,
            )
        return engine.run()

    for _ in range(warmup):
        run_static_workload()
        run_continuous_workload()

    static_results = []
    continuous_results = []
    for iteration in range(iterations):
        # 交替先后顺序，降低温度、频率变化对某一模式的系统性偏置。
        if iteration % 2 == 0:
            static_results.append(run_static_workload())
            continuous_results.append(run_continuous_workload())
        else:
            continuous_results.append(run_continuous_workload())
            static_results.append(run_static_workload())

    static_seconds = sum(result["seconds"] for result in static_results)
    static_tokens = sum(result["tokens"] for result in static_results)
    continuous_seconds = sum(
        result.total_seconds for result in continuous_results
    )
    continuous_tokens = sum(
        result.total_output_tokens for result in continuous_results
    )
    static_tps = static_tokens / static_seconds
    continuous_tps = continuous_tokens / continuous_seconds
    speedup = continuous_tps / static_tps

    metrics = {
        "backend": "continuous",
        "max_num_seqs": args.batch_size,
        "num_requests": args.num_requests,
        "iterations": iterations,
        "static_tokens_per_second": static_tps,
        "continuous_tokens_per_second": continuous_tps,
        "speedup": speedup,
        "static_requests_per_second": (
            args.num_requests * iterations / static_seconds
        ),
        "continuous_requests_per_second": (
            args.num_requests * iterations / continuous_seconds
        ),
        "static_peak_memory_mib": max(
            result["peak_memory_mib"] for result in static_results
        ),
        "continuous_peak_memory_mib": max(
            result.peak_memory_mib for result in continuous_results
        ),
        "static_mean_request_latency_ms": statistics.fmean(
            seconds * 1000
            for result in static_results
            for seconds in result["completion_seconds"]
        ),
        "continuous_mean_request_latency_ms": statistics.fmean(
            seconds * 1000
            for result in continuous_results
            for seconds in result.request_finish_seconds
        ),
    }

    print("Scheduler workload: static vs continuous")
    print(f"  请求数         : {args.num_requests}")
    print(f"  最大并发       : {args.batch_size}")
    print(f"  生成上限       : {limits}")
    print(f"  Static 吞吐    : {static_tps:.2f} tokens/s")
    print(f"  Continuous 吞吐: {continuous_tps:.2f} tokens/s")
    print(f"  吞吐倍率       : {speedup:.2f}x")
    print(
        "  平均请求延迟   : "
        f"{metrics['static_mean_request_latency_ms']:.2f} ms static / "
        f"{metrics['continuous_mean_request_latency_ms']:.2f} ms continuous"
    )
    print(
        "  Static/Continuous 峰值显存: "
        f"{metrics['static_peak_memory_mib']:.1f} / "
        f"{metrics['continuous_peak_memory_mib']:.1f} MiB"
    )

    if args.save:
        append_record(
            args.save,
            label=label,
            metrics=metrics,
            metadata={
                "model": args.model,
                "prompt": args.prompt,
                "prompt_tokens": len(prompt_ids),
                "request_max_new_tokens": limits,
                "warmup": warmup,
                "decoding": (
                    "greedy" if sampling_params.is_greedy else "sampling"
                ),
            },
        )
        print(f"\n结果已追加到：{args.save}")


def run_static_batch_benchmark(
    args: argparse.Namespace,
    tokenizer: Tokenizer,
) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size 必须大于 0")
    warmup = 1 if args.warmup is None else args.warmup
    iterations = 3 if args.iterations is None else args.iterations
    label = args.label or f"stage-07-static-batch-{args.batch_size}"
    config = ModelConfig.from_pretrained(args.model)
    prompt_ids = tokenizer.encode_chat(
        [{"role": "user", "content": args.prompt}],
        enable_thinking=False,
    )
    prompts = [prompt_ids] * args.batch_size
    loaded = load_model(config, device="cuda")

    sampling_params = resolve_sampling_params(args, config)

    def generate() -> BatchGenerationResult:
        return generate_static_batch(
            loaded.model,
            prompts,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=set(config.generation_eos_token_ids),
            pad_token_id=tokenizer.pad_token_id,
            sampling_params=sampling_params,
        )

    for _ in range(warmup):
        generate()
    results = [generate() for _ in range(iterations)]
    total_tokens = sum(result.total_output_tokens for result in results)
    total_seconds = sum(sum(result.step_seconds) for result in results)
    total_requests = args.batch_size * iterations
    strategy = "greedy" if sampling_params.is_greedy else "sampling"
    metrics = {
        "backend": "static",
        "batch_size": args.batch_size,
        "iterations": iterations,
        "mean_ttft_ms": statistics.fmean(result.ttft_ms for result in results),
        "mean_tpot_ms": statistics.fmean(result.tpot_ms for result in results),
        "output_tokens_per_second": total_tokens / total_seconds,
        "requests_per_second": total_requests / total_seconds,
        "peak_memory_mib": max(result.peak_memory_mib for result in results),
        "model_load_seconds": loaded.load_seconds,
    }

    print(f"Static batch {strategy} generation")
    print(f"  Batch size : {args.batch_size}")
    print(f"  模型加载   : {metrics['model_load_seconds']:.2f} s")
    print(f"  平均 TTFT  : {metrics['mean_ttft_ms']:.2f} ms")
    print(f"  每轮 TPOT  : {metrics['mean_tpot_ms']:.2f} ms")
    print(f"  总输出吞吐 : {metrics['output_tokens_per_second']:.2f} tokens/s")
    print(f"  请求吞吐   : {metrics['requests_per_second']:.2f} requests/s")
    print(f"  峰值显存   : {metrics['peak_memory_mib']:.1f} MiB")

    if args.save:
        append_record(
            args.save,
            label=label,
            metrics=metrics,
            metadata={
                "model": args.model,
                "prompt": args.prompt,
                "prompt_tokens": len(prompt_ids),
                "max_new_tokens": args.max_new_tokens,
                "warmup": warmup,
                "decoding": strategy,
            },
        )
        print(f"\n结果已追加到：{args.save}")


def run_generation_benchmark(
    args: argparse.Namespace,
    tokenizer: Tokenizer,
) -> None:
    warmup = 1 if args.warmup is None else args.warmup
    iterations = 3 if args.iterations is None else args.iterations
    config = ModelConfig.from_pretrained(args.model)
    if args.prompt_repeat <= 0:
        raise ValueError("--prompt-repeat 必须大于 0")
    prompt_text = args.prompt * args.prompt_repeat
    messages: list[ChatMessage] = [{"role": "user", "content": prompt_text}]
    prompt_token_ids = tokenizer.encode_chat(messages, enable_thinking=False)
    loaded = load_model(config, device="cuda")
    sampling_params = resolve_sampling_params(args, config)
    label = args.label or (
        "stage-06-sampling"
        if not sampling_params.is_greedy
        else (
            "stage-05-kv-cache"
            if args.backend == "cached"
            else "stage-04-naive"
        )
    )
    generate_function = (
        generate_greedy_cached
        if args.backend == "cached"
        else generate_greedy_naive
    )

    def generate() -> GenerationResult:
        return generate_function(
            loaded.model,
            prompt_token_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=set(config.generation_eos_token_ids),
            sampling_params=sampling_params,
        )

    for _ in range(warmup):
        generate()
    results = [generate() for _ in range(iterations)]

    total_tokens = sum(len(result.output_token_ids) for result in results)
    total_seconds = sum(sum(result.step_seconds) for result in results)
    metrics = {
        "backend": args.backend,
        "iterations": iterations,
        "mean_ttft_ms": statistics.fmean(result.ttft_ms for result in results),
        "mean_tpot_ms": statistics.fmean(result.tpot_ms for result in results),
        "output_tokens_per_second": total_tokens / total_seconds,
        "decode_tokens_per_second": statistics.fmean(
            result.decode_tokens_per_second for result in results
        ),
        "peak_memory_mib": max(result.peak_memory_mib for result in results),
        "model_load_seconds": loaded.load_seconds,
    }
    strategy = "greedy" if sampling_params.is_greedy else "sampling"
    print(f"{args.backend.capitalize()} {strategy} generation")
    print(f"  模型加载 : {metrics['model_load_seconds']:.2f} s")
    print(f"  平均 TTFT: {metrics['mean_ttft_ms']:.2f} ms")
    print(f"  平均 TPOT: {metrics['mean_tpot_ms']:.2f} ms")
    print(f"  输出吞吐 : {metrics['output_tokens_per_second']:.2f} tokens/s")
    print(f"  Decode吞吐: {metrics['decode_tokens_per_second']:.2f} tokens/s")
    print(f"  峰值显存 : {metrics['peak_memory_mib']:.1f} MiB")

    if args.save:
        append_record(
            args.save,
            label=label,
            metrics=metrics,
            metadata={
                "model": args.model,
                "prompt": args.prompt,
                "prompt_repeat": args.prompt_repeat,
                "prompt_tokens": len(prompt_token_ids),
                "max_new_tokens": args.max_new_tokens,
                "warmup": warmup,
                "decoding": strategy,
                "sampling": {
                    "temperature": sampling_params.temperature,
                    "top_k": sampling_params.top_k,
                    "top_p": sampling_params.top_p,
                    "seed": sampling_params.seed,
                },
            },
        )
        print(f"\n结果已追加到：{args.save}")


if __name__ == "__main__":
    main()

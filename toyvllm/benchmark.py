from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class BenchmarkResult:
    """统一保存 benchmark 指标，后续会继续加入 TTFT、TPOT 和显存峰值。"""

    name: str
    iterations: int
    latencies_ms: list[float] = field(repr=False)
    items_per_iteration: int = 1
    unit: str = "items"

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.latencies_ms)

    @property
    def p50_ms(self) -> float:
        return statistics.median(self.latencies_ms)

    @property
    def min_ms(self) -> float:
        return min(self.latencies_ms)

    @property
    def throughput(self) -> float:
        total_items = self.items_per_iteration * self.iterations
        total_seconds = sum(self.latencies_ms) / 1000
        return total_items / total_seconds

    def format(self) -> str:
        return (
            f"{self.name}\n"
            f"  平均时延 : {self.mean_ms:.3f} ms\n"
            f"  P50 时延 : {self.p50_ms:.3f} ms\n"
            f"  最低时延 : {self.min_ms:.3f} ms\n"
            f"  吞吐量   : {self.throughput:.2f} {self.unit}/s"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "items_per_iteration": self.items_per_iteration,
            "unit": self.unit,
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms,
            "min_ms": self.min_ms,
            "throughput": self.throughput,
        }


def append_result(
    output_path: str | Path,
    result: BenchmarkResult,
    *,
    label: str,
    metadata: dict[str, object] | None = None,
) -> None:
    """按 JSON Lines 格式追加结果，保留每个演进阶段的历史数据。

    每次 benchmark 写一行独立 JSON。追加新指标时不必改旧记录，也方便后续用
    Python、Excel 或其他工具读取并画图。
    """

    append_record(
        output_path,
        label=label,
        metrics=result.to_dict(),
        metadata=metadata,
    )


def append_record(
    output_path: str | Path,
    *,
    label: str,
    metrics: dict[str, object],
    metadata: dict[str, object] | None = None,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "label": label,
        **metrics,
        "metadata": metadata or {},
    }
    with path.open("a", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False)
        file.write("\n")


def benchmark(
    name: str,
    operation: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
    items_per_iteration: int = 1,
    unit: str = "items",
) -> BenchmarkResult:
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup 不能为负数，iterations 必须大于 0")

    for _ in range(warmup):
        operation()

    latencies_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        operation()
        latencies_ms.append((time.perf_counter() - start) * 1000)

    return BenchmarkResult(
        name=name,
        iterations=iterations,
        latencies_ms=latencies_ms,
        items_per_iteration=items_per_iteration,
        unit=unit,
    )

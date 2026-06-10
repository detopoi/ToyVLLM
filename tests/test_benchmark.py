import json
import tempfile
import unittest
from pathlib import Path

from toyvllm.benchmark import BenchmarkResult, append_result, percentile


class BenchmarkTest(unittest.TestCase):
    def test_percentile_uses_linear_interpolation(self) -> None:
        values = [10.0, 20.0, 30.0, 40.0]
        self.assertEqual(percentile(values, 0.0), 10.0)
        self.assertEqual(percentile(values, 0.5), 25.0)
        self.assertEqual(percentile(values, 0.95), 38.5)

    def test_metrics(self) -> None:
        result = BenchmarkResult(
            name="test",
            iterations=2,
            latencies_ms=[10.0, 20.0],
            items_per_iteration=3,
            unit="tokens",
        )
        self.assertEqual(result.mean_ms, 15.0)
        self.assertEqual(result.p50_ms, 15.0)
        self.assertAlmostEqual(result.throughput, 200.0)

    def test_append_jsonl_result(self) -> None:
        result = BenchmarkResult(
            name="test",
            iterations=1,
            latencies_ms=[10.0],
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            append_result(output, result, label="stage-test")
            record = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(record["label"], "stage-test")
        self.assertEqual(record["mean_ms"], 10.0)


if __name__ == "__main__":
    unittest.main()

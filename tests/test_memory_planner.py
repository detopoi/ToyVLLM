import unittest

from toyvllm.engine import calculate_kv_cache_capacity


MIB = 1024**2


class MemoryPlannerTest(unittest.TestCase):
    def test_calculates_blocks_from_utilization_target(self) -> None:
        plan = calculate_kv_cache_capacity(
            total_memory_bytes=8_000 * MIB,
            free_memory_bytes=4_500 * MIB,
            gpu_memory_utilization=0.85,
            runtime_reserve_bytes=1_000 * MIB,
            bytes_per_block=100 * MIB,
        )

        # target=6800 MiB, current=3500 MiB, reserve=1000 MiB，
        # 剩余 2300 MiB，恰好可放 23 个 Block。
        self.assertEqual(plan.available_cache_bytes, 2_300 * MIB)
        self.assertEqual(plan.num_blocks, 23)
        self.assertEqual(plan.current_used_bytes, 3_500 * MIB)

    def test_workspace_bytes_are_charged_per_block(self) -> None:
        plan = calculate_kv_cache_capacity(
            total_memory_bytes=1_000,
            free_memory_bytes=1_000,
            gpu_memory_utilization=1.0,
            runtime_reserve_bytes=0,
            bytes_per_block=90,
            workspace_bytes_per_block=10,
        )
        self.assertEqual(plan.num_blocks, 10)
        self.assertEqual(plan.allocated_cache_bytes, 900)

    def test_negative_budget_produces_zero_blocks(self) -> None:
        plan = calculate_kv_cache_capacity(
            total_memory_bytes=1_000,
            free_memory_bytes=100,
            gpu_memory_utilization=0.8,
            runtime_reserve_bytes=100,
            bytes_per_block=50,
        )
        self.assertEqual(plan.available_cache_bytes, 0)
        self.assertEqual(plan.num_blocks, 0)

    def test_invalid_utilization_is_rejected(self) -> None:
        for utilization in (0.0, 1.1):
            with self.assertRaises(ValueError):
                calculate_kv_cache_capacity(
                    total_memory_bytes=1_000,
                    free_memory_bytes=500,
                    gpu_memory_utilization=utilization,
                    runtime_reserve_bytes=0,
                    bytes_per_block=100,
                )


if __name__ == "__main__":
    unittest.main()

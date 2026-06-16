import unittest

from ai.batch_evaluator_benchmark import (
    benchmark_batch_evaluators,
    format_benchmark_results,
    generate_benchmark_boards,
)


class TestBatchEvaluatorBenchmark(unittest.TestCase):
    def test_generate_benchmark_boards_is_deterministic(self):
        first = generate_benchmark_boards(positions := 6, seed=7)
        second = generate_benchmark_boards(positions, seed=7)

        self.assertEqual(
            [board.grid for board in first],
            [board.grid for board in second],
        )

    def test_python_and_cpu_vector_scores_match(self):
        results = benchmark_batch_evaluators(
            ["python", "cpu-vector"],
            position_count=6,
            seed=3,
        )

        self.assertEqual(["python", "cpu-vector"], [result.backend for result in results])
        self.assertEqual(0, results[1].mismatch_count)
        self.assertEqual(0.0, results[1].max_delta_from_python)

    def test_format_benchmark_results_includes_backends(self):
        results = benchmark_batch_evaluators(
            ["python", "cpu-vector"],
            position_count=2,
            seed=1,
        )

        output = format_benchmark_results(results)

        self.assertIn("backend", output)
        self.assertIn("python", output)
        self.assertIn("cpu-vector", output)


if __name__ == "__main__":
    unittest.main()

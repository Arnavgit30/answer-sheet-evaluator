import json
import math
import tempfile
import unittest
from pathlib import Path

from backend.app.benchmarking import benchmark_grading, compute_metrics


class BenchmarkTests(unittest.TestCase):
    def test_metrics_for_known_pairs(self):
        metrics = compute_metrics(
            [
                {"question_id": "Q1", "predicted_marks": 5, "human_marks": 4},
                {"question_id": "Q2", "predicted_marks": 2, "human_marks": 2},
                {"question_id": "Q3", "predicted_marks": 0, "human_marks": 1},
            ]
        )
        self.assertEqual(metrics["status"], "ground_truth_available")
        self.assertAlmostEqual(metrics["mae"], 2 / 3)
        self.assertAlmostEqual(metrics["rmse"], math.sqrt(2 / 3))
        self.assertAlmostEqual(metrics["agreement_within_1_mark"], 1.0)
        self.assertIsNotNone(metrics["pearson_correlation"])
        self.assertIsNotNone(metrics["spearman_correlation"])

    def test_metrics_are_null_without_human_marks(self):
        metrics = compute_metrics([])
        self.assertEqual(metrics["status"], "ground_truth_unavailable")
        self.assertIsNone(metrics["mae"])
        self.assertIsNone(metrics["pearson_correlation"])

    def test_benchmark_result_is_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            grading = root / "grading.json"
            human = root / "human.json"
            grading.write_text(json.dumps({"document_id": "student_001", "questions": [{"question_id": "Q1", "predicted_marks": 4}]}), encoding="utf-8")
            human.write_text(json.dumps({"marks": [{"question_id": "Q1", "human_marks": 5}]}), encoding="utf-8")
            first = benchmark_grading(grading, human, root / "benchmark")
            second = benchmark_grading(grading, human, root / "benchmark")
        self.assertFalse(first.cached)
        self.assertTrue(second.cached)


if __name__ == "__main__":
    unittest.main()

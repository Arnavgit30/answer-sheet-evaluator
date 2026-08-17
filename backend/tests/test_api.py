import unittest

from fastapi.testclient import TestClient

from backend.app.main import create_app


class FakeEvaluationService:
    def validate_dataset(self, config_path):
        return {"valid": True, "errors": [], "manifest": {"dataset_id": "demo", "student_documents": [], "answer_keys": [], "human_marks_available": False}}

    def start_evaluation(self, config_path, student_document_id):
        return {
            "evaluation_id": "evaluation_001",
            "status": "completed",
            "dataset_id": "demo",
            "document_id": "student_001",
            "answer_key_id": "key_001",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:01+00:00",
            "error": None,
        }

    def get_evaluation(self, evaluation_id):
        return self.start_evaluation("", None) if evaluation_id == "evaluation_001" else None

    def get_results(self, evaluation_id):
        if evaluation_id != "evaluation_001":
            return None
        return {
            "evaluation_id": evaluation_id,
            "status": "completed",
            "grading": {"grading_method": "baseline_similarity", "questions": [], "total_predicted_marks": 0, "total_maximum_marks": 0},
            "benchmark": {"status": "ground_truth_unavailable"},
        }


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app(FakeEvaluationService()))

    def test_health(self):
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_validate_dataset(self):
        response = self.client.post("/api/v1/datasets/validate", json={"config_path": "configs/demo.yaml"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["valid"])

    def test_evaluation_status_and_results(self):
        created = self.client.post("/api/v1/evaluations", json={}).json()
        status = self.client.get(f"/api/v1/evaluations/{created['evaluation_id']}")
        results = self.client.get(f"/api/v1/evaluations/{created['evaluation_id']}/results")
        self.assertEqual(status.status_code, 200)
        self.assertEqual(results.status_code, 200)
        self.assertEqual(results.json()["grading"]["grading_method"], "baseline_similarity")


if __name__ == "__main__":
    unittest.main()

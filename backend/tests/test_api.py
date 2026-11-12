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

    def get_questions(self, evaluation_id):
        if evaluation_id != "evaluation_001":
            return []
        return [{"question_id": "Q1", "student_answer": {"text": "Ans"}}]

    def get_question_detail(self, evaluation_id, question_id):
        if evaluation_id == "evaluation_001" and question_id == "Q1":
            return {"question_id": "Q1", "student_answer": {"text": "Ans"}}
        return None

    def get_ocr_reviews(self, evaluation_id):
        if evaluation_id != "evaluation_001":
            return []
        return [{"review_id": "rev_ocr_1", "question_id": "Q1", "reviewer_action": "pending"}]

    def update_ocr_review(self, evaluation_id, review_id, action, corrected_text=None):
        if evaluation_id == "evaluation_001" and review_id == "rev_ocr_1":
            return {"review_id": review_id, "reviewer_action": action, "corrected_text": corrected_text}
        return None

    def get_grading_reviews(self, evaluation_id):
        if evaluation_id != "evaluation_001":
            return []
        return [{"review_id": "rev_gr_1", "question_id": "Q1", "reviewer_action": "pending"}]

    def update_grading_review(self, evaluation_id, review_id, action, human_marks=None, notes=None):
        if evaluation_id == "evaluation_001" and review_id == "rev_gr_1":
            return {"review_id": review_id, "reviewer_action": action, "human_marks": human_marks}
        return None

    def get_page_image_path(self, evaluation_id, page_number):
        return None


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

    def test_question_detail_endpoints(self):
        res = self.client.get("/api/v1/evaluations/evaluation_001/questions")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()), 1)
        self.assertEqual(res.json()[0]["question_id"], "Q1")

        q1 = self.client.get("/api/v1/evaluations/evaluation_001/questions/Q1")
        self.assertEqual(q1.status_code, 200)
        self.assertEqual(q1.json()["question_id"], "Q1")

    def test_ocr_review_endpoints(self):
        revs = self.client.get("/api/v1/evaluations/evaluation_001/ocr-reviews")
        self.assertEqual(revs.status_code, 200)
        self.assertEqual(len(revs.json()), 1)

        updated = self.client.post(
            "/api/v1/evaluations/evaluation_001/ocr-reviews/rev_ocr_1",
            json={"action": "corrected", "corrected_text": "Fixed text"},
        )
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json()["action" if "action" in updated.json() else "reviewer_action"], "corrected")

    def test_grading_review_endpoints(self):
        revs = self.client.get("/api/v1/evaluations/evaluation_001/grading-reviews")
        self.assertEqual(revs.status_code, 200)
        self.assertEqual(len(revs.json()), 1)

        updated = self.client.post(
            "/api/v1/evaluations/evaluation_001/grading-reviews/rev_gr_1",
            json={"action": "overridden", "human_marks": 4.5},
        )
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json()["human_marks"], 4.5)


if __name__ == "__main__":
    unittest.main()

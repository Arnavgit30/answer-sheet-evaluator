"""Unit tests for vlm_evaluator."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.app.models import Formula, ReferenceAnswer, StudentAnswer, Table
from backend.app.vlm_evaluator import VLMEvaluator


def _mock_llm_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {
        "choices": [{"message": {"content": json.dumps(payload)}}]
    }
    resp.raise_for_status.return_value = None
    return resp


class VLMEvaluatorTests(unittest.TestCase):
    def test_correct_answer_receives_appropriate_marks(self):
        evaluator = VLMEvaluator(api_key="fake-key", model="qwen/qwen3.8-27b")
        student = StudentAnswer(
            question_id="Q1",
            text="Photosynthesis converts solar energy into chemical energy stored in glucose molecules.",
        )
        reference = ReferenceAnswer(
            question_id="Q1",
            answer_text="Photosynthesis converts light energy into chemical energy stored in glucose.",
            maximum_marks=5.0,
            rubric="2.5 marks for energy conversion, 2.5 marks for glucose storage.",
            key_concepts=["light energy", "chemical energy", "glucose"],
        )

        fake_resp = _mock_llm_response(
            {
                "marks_awarded": 5.0,
                "reasoning": "Complete and accurate explanation of energy conversion and glucose storage.",
                "concepts_present": ["light energy", "chemical energy", "glucose"],
                "concepts_missing": [],
                "errors": [],
                "grading_confidence": 0.96,
                "human_review_required": False,
            }
        )

        with patch("requests.post", return_value=fake_resp):
            result = evaluator.evaluate_question(student, reference)

        self.assertEqual(result.marks_awarded, 5.0)
        self.assertEqual(result.max_marks, 5.0)
        self.assertEqual(result.status, "completed")
        self.assertFalse(result.human_review_required)
        self.assertEqual(result.grading_confidence, 0.96)
        self.assertEqual(result.grading_method, "qwen/qwen3.8-27b")

    def test_partially_correct_answer_receives_partial_marks(self):
        evaluator = VLMEvaluator(api_key="fake-key")
        student = StudentAnswer(question_id="Q2", text="Water turns into vapour.")
        reference = ReferenceAnswer(
            question_id="Q2",
            answer_text="Evaporation is the change of liquid water into water vapour caused by heat.",
            maximum_marks=4.0,
        )

        fake_resp = _mock_llm_response(
            {
                "marks_awarded": 2.0,
                "reasoning": "Identifies phase change to vapour but omits the role of heat energy.",
                "concepts_present": ["phase change to vapour"],
                "concepts_missing": ["heat energy driver"],
                "errors": [],
                "grading_confidence": 0.88,
                "human_review_required": False,
            }
        )

        with patch("requests.post", return_value=fake_resp):
            result = evaluator.evaluate_question(student, reference)

        self.assertEqual(result.marks_awarded, 2.0)
        self.assertGreater(result.marks_awarded, 0.0)
        self.assertLess(result.marks_awarded, result.max_marks)
        self.assertFalse(result.human_review_required)

    def test_incorrect_answer_receives_zero_marks(self):
        evaluator = VLMEvaluator(api_key="fake-key")
        student = StudentAnswer(question_id="Q3", text="The moon is made of cheese.")
        reference = ReferenceAnswer(
            question_id="Q3",
            answer_text="An acid turns blue litmus paper red.",
            maximum_marks=6.0,
        )

        fake_resp = _mock_llm_response(
            {
                "marks_awarded": 0.0,
                "reasoning": "Answer is completely unrelated to acid-base indicators.",
                "concepts_present": [],
                "concepts_missing": ["acid litmus test", "base litmus test"],
                "errors": ["Irrelevant response"],
                "grading_confidence": 0.99,
                "human_review_required": False,
            }
        )

        with patch("requests.post", return_value=fake_resp):
            result = evaluator.evaluate_question(student, reference)

        self.assertEqual(result.marks_awarded, 0.0)
        self.assertEqual(result.status, "completed")
        self.assertFalse(result.human_review_required)

    def test_valid_alternative_solution_is_not_penalized_by_low_similarity(self):
        evaluator = VLMEvaluator(api_key="fake-key")
        student = StudentAnswer(
            question_id="Q4",
            text="Using conservation of mechanical energy: mgh = 1/2 mv^2, so v = sqrt(2gh).",
            formulas=[Formula("Q4_F1", "v = sqrt(2gh)", "r1", 1)],
            tables=[Table("Q4_T1", "h | v", "r2", 1)],
        )
        reference = ReferenceAnswer(
            question_id="Q4",
            answer_text="Apply kinematic equation v^2 = u^2 + 2as with u=0 and a=g to obtain v = sqrt(2gh).",
            maximum_marks=5.0,
        )

        fake_resp = _mock_llm_response(
            {
                "marks_awarded": 5.0,
                "reasoning": "Valid alternative derivation using conservation of energy instead of kinematics.",
                "concepts_present": ["energy conservation", "final velocity derivation"],
                "concepts_missing": [],
                "errors": [],
                "grading_confidence": 0.94,
                "human_review_required": False,
            }
        )

        with patch("requests.post", return_value=fake_resp):
            result = evaluator.evaluate_question(student, reference, baseline_similarity=18.0)

        self.assertEqual(result.marks_awarded, 5.0)
        self.assertEqual(result.baseline_similarity, 18.0)
        self.assertIsNone(result.baseline_marks)
        self.assertFalse(result.human_review_required)

    def test_blank_answer_awards_zero_and_flags_human_review(self):
        evaluator = VLMEvaluator(api_key="fake-key")
        student = StudentAnswer(question_id="Q1", text="   ")
        reference = ReferenceAnswer(question_id="Q1", answer_text="Ref text", maximum_marks=5.0)

        with patch("requests.post") as mock_post:
            result = evaluator.evaluate_question(student, reference)
            mock_post.assert_not_called()

        self.assertEqual(result.marks_awarded, 0.0)
        self.assertTrue(result.human_review_required)
        self.assertIn("human review required", result.reasoning.lower())

    def test_missing_api_key_marks_pending_and_flags_human_review_without_fallback(self):
        evaluator = VLMEvaluator(api_key="")
        student = StudentAnswer(question_id="Q1", text="Student answer text")
        reference = ReferenceAnswer(question_id="Q1", answer_text="Ref text", maximum_marks=5.0)

        result = evaluator.evaluate_question(student, reference, baseline_similarity=85.0, baseline_marks=4.5)
        self.assertEqual(result.question_id, "Q1")
        self.assertIsNone(result.marks_awarded)
        self.assertEqual(result.status, "pending")
        self.assertTrue(result.human_review_required)
        self.assertEqual(result.grading_method, "llm_pending")
        self.assertIn("Missing VLM_API_KEY", result.errors)

    def test_api_failure_flags_human_review_with_no_baseline_fallback(self):
        evaluator = VLMEvaluator(api_key="fake-key")
        student = StudentAnswer(question_id="Q1", text="Student answer")
        reference = ReferenceAnswer(question_id="Q1", answer_text="Ref answer", maximum_marks=5.0)

        with patch("requests.post", side_effect=RuntimeError("Connection refused")):
            result = evaluator.evaluate_question(student, reference, baseline_similarity=90.0, baseline_marks=4.5)

        self.assertIsNone(result.marks_awarded)
        self.assertEqual(result.status, "failed")
        self.assertTrue(result.human_review_required)
        self.assertEqual(result.grading_method, "llm_failed")
        self.assertIn("Connection refused", result.reasoning)

    def test_invalid_llm_json_is_safely_rejected(self):
        evaluator = VLMEvaluator(api_key="fake-key")
        student = StudentAnswer(question_id="Q1", text="Student answer")
        reference = ReferenceAnswer(question_id="Q1", answer_text="Ref answer", maximum_marks=5.0)

        fake_resp = MagicMock()
        fake_resp.json.return_value = {"choices": [{"message": {"content": "Not valid JSON at all"}}]}
        fake_resp.raise_for_status.return_value = None

        with patch("requests.post", return_value=fake_resp):
            result = evaluator.evaluate_question(student, reference)

        self.assertIsNone(result.marks_awarded)
        self.assertEqual(result.status, "failed")
        self.assertTrue(result.human_review_required)

    def test_marks_exceeding_maximum_are_rejected(self):
        evaluator = VLMEvaluator(api_key="fake-key")
        student = StudentAnswer(question_id="Q1", text="Student answer")
        reference = ReferenceAnswer(question_id="Q1", answer_text="Ref answer", maximum_marks=5.0)

        fake_resp = _mock_llm_response(
            {
                "marks_awarded": 7.5,  # Exceeds max_marks=5.0!
                "reasoning": "Great answer.",
                "concepts_present": [],
                "concepts_missing": [],
                "errors": [],
                "grading_confidence": 0.9,
                "human_review_required": False,
            }
        )

        with patch("requests.post", return_value=fake_resp):
            result = evaluator.evaluate_question(student, reference)

        self.assertIsNone(result.marks_awarded)
        self.assertEqual(result.status, "failed")
        self.assertTrue(result.human_review_required)
        self.assertTrue(any("out of valid range" in e for e in result.errors))

    def test_sends_ocr_extracted_text_only_without_base64_images(self):
        evaluator = VLMEvaluator(api_key="fake-key")
        with tempfile.TemporaryDirectory() as tmpdir:
            img_file = Path(tmpdir) / "page_001.png"
            img_file.write_bytes(b"\x89PNG\r\n\x1a\nfakeimage")
            student = StudentAnswer(
                question_id="Q1",
                text="Handwritten answer with diagram text extracted by OCR",
                answer_images=[str(img_file)],
            )
            reference = ReferenceAnswer(question_id="Q1", answer_text="Ref", maximum_marks=5.0)
            fake_resp = _mock_llm_response(
                {
                    "marks_awarded": 4.0,
                    "reasoning": "Diagram and text mostly accurate.",
                    "concepts_present": ["diagram"],
                    "concepts_missing": [],
                    "errors": [],
                    "grading_confidence": 0.9,
                    "human_review_required": False,
                }
            )
            with patch("requests.post", return_value=fake_resp) as mock_post:
                result = evaluator.evaluate_question(student, reference)
                sent_json = mock_post.call_args.kwargs["json"]
                user_content = sent_json["messages"][1]["content"]

        self.assertEqual(result.marks_awarded, 4.0)
        self.assertIsInstance(user_content, str)
        self.assertIn("Handwritten answer with diagram text extracted by OCR", user_content)


if __name__ == "__main__":
    unittest.main()

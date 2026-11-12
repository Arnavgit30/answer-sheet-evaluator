import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.app.answer_key_parser import parse_answer_key_file
from backend.app.grading import LLMGrader, grade_from_files, normalize_text
from backend.app.models import EvaluationResult
from backend.app.segmentation import segment_normalized_document
from backend.app.vlm_evaluator import VLMEvaluator


ROOT = Path(__file__).resolve().parents[2]
KEY_FIXTURE = ROOT / "data" / "synthetic" / "answer_key.json"
STUDENT_FIXTURE = ROOT / "data" / "synthetic" / "student_normalized.json"


class FakeLLMGrader(LLMGrader):
    """Deterministic LLM mock grader for fixture pipeline testing."""

    def __init__(self, score_map: dict[str, float] | None = None) -> None:
        super().__init__(evaluator=VLMEvaluator(api_key="test-key", model="qwen/qwen3.8-27b"))
        self.score_map = score_map or {"Q1": 4.5, "Q2": 3.0, "Q3": 5.5}

    def _grade_question(self, question_id, student, key):
        def _fake_eval(student_obj, ref_obj, baseline_similarity=None, baseline_marks=None):
            if not (student_obj.text or "").strip():
                return EvaluationResult(
                    question_id=question_id,
                    marks_awarded=0.0,
                    max_marks=float(ref_obj.maximum_marks or 5.0),
                    reasoning="No student answer detected in segmented regions — human review required!",
                    grading_confidence=0.0,
                    grading_method="qwen/qwen3.8-27b",
                    baseline_similarity=baseline_similarity,
                    human_review_required=True,
                    status="completed",
                )
            awarded = min(self.score_map.get(question_id, 0.0), float(ref_obj.maximum_marks or 5.0))
            return EvaluationResult(
                question_id=question_id,
                marks_awarded=awarded,
                max_marks=float(ref_obj.maximum_marks or 5.0),
                reasoning=f"Evaluated {question_id} conceptually via LLM.",
                concepts_present=["core concept"],
                concepts_missing=[],
                errors=[],
                grading_confidence=0.92,
                grading_method="qwen/qwen3.8-27b",
                baseline_similarity=baseline_similarity,
                human_review_required=False,
                status="completed",
            )

        with patch.object(self.evaluator, "evaluate_question", side_effect=_fake_eval):
            return super()._grade_question(question_id, student, key)


class LLMGradingTests(unittest.TestCase):
    def grade_fixture(self, replacements=None, score_map=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            student = json.loads(STUDENT_FIXTURE.read_text(encoding="utf-8"))
            for page in student["pages"]:
                for element in page["elements"]:
                    if replacements and element["text"][:2] in replacements:
                        element["text"] = replacements[element["text"][:2]]
            student_file = root / "student.json"
            student_file.write_text(json.dumps(student), encoding="utf-8")
            segments = segment_normalized_document(student_file, root / "segments")
            key = parse_answer_key_file(KEY_FIXTURE, root / "keys")
            grader = FakeLLMGrader(score_map=score_map)
            result = grade_from_files(
                segments.segmentation_path, key.answer_key_path, root / "grading", grader=grader
            )
            return json.loads(Path(result.grading_path).read_text(encoding="utf-8"))

    def test_correct_answer_receives_llm_mark_not_similarity_mark(self):
        result = self.grade_fixture(score_map={"Q1": 5.0, "Q2": 4.0, "Q3": 6.0})
        q1 = result["questions"][0]
        self.assertEqual(q1["predicted_marks"], 5.0)
        self.assertEqual(q1["grading_method"], "qwen/qwen3.8-27b")
        self.assertEqual(result["total_predicted_marks"], 15.0)
        self.assertEqual(result["total_maximum_marks"], 15.0)

    def test_partially_correct_answer_receives_partial_llm_mark(self):
        result = self.grade_fixture(
            {"Q2": "Q2 Water changes into vapour."},
            score_map={"Q1": 4.5, "Q2": 2.0, "Q3": 5.0},
        )
        q2 = result["questions"][1]
        self.assertEqual(q2["predicted_marks"], 2.0)
        self.assertGreater(q2["predicted_marks"], 0)
        self.assertLess(q2["predicted_marks"], q2["maximum_marks"])

    def test_unrelated_answer_receives_zero_llm_mark(self):
        result = self.grade_fixture(
            {"Q3": "Q3(a) The moon is made of cheese."},
            score_map={"Q1": 4.5, "Q2": 3.0, "Q3": 0.0},
        )
        q3 = result["questions"][2]
        self.assertEqual(q3["predicted_marks"], 0.0)
        self.assertLess(q3["similarity_score"], 50)

    def test_missing_answer_is_scored_zero_and_flagged_for_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = parse_answer_key_file(KEY_FIXTURE, root / "keys")
            student = root / "segments.json"
            student.write_text(json.dumps({"document_id": "student_001", "questions": []}), encoding="utf-8")
            result = grade_from_files(
                student, key.answer_key_path, root / "grading", grader=FakeLLMGrader()
            )
            payload = json.loads(Path(result.grading_path).read_text(encoding="utf-8"))
        self.assertTrue(all(question["predicted_marks"] == 0.0 for question in payload["questions"]))
        self.assertTrue(all(question["human_review_required"] is True for question in payload["questions"]))

    def test_text_normalization_is_deterministic(self):
        self.assertEqual(normalize_text(" Light-energy,  GLUCOSE! "), "light energy glucose")


if __name__ == "__main__":
    unittest.main()

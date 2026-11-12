"""Unit tests for grading_review."""

import unittest

from backend.app.grading_review import (
    apply_grading_override,
    create_grading_reviews,
    evaluate_grading_confidence,
)
from backend.app.models import EvaluationResult, GradingReview


class GradingReviewTests(unittest.TestCase):
    def test_low_confidence_when_reasoning_brief(self):
        res = EvaluationResult(
            question_id="Q1",
            marks_awarded=2.0,
            max_marks=5.0,
            reasoning="Ok",
        )
        conf = evaluate_grading_confidence(res)
        self.assertLess(conf, 0.7)

    def test_valid_alternative_solution_not_penalized_by_low_baseline_similarity(self):
        res = EvaluationResult(
            question_id="Q1",
            marks_awarded=5.0,
            max_marks=5.0,
            reasoning="A thorough detailed alternative solution covering all concepts completely.",
            grading_confidence=0.92,
            baseline_similarity=10.0,
        )
        conf = evaluate_grading_confidence(res)
        self.assertGreaterEqual(conf, 0.9)
        reviews = create_grading_reviews("eval_123", [res], confidence_threshold=0.7)
        self.assertEqual(len(reviews), 0)

    def test_failed_or_pending_evaluation_is_flagged_for_human_review(self):
        failed = EvaluationResult(
            question_id="Q1",
            marks_awarded=None,
            max_marks=5.0,
            reasoning="LLM evaluation failed (timeout).",
            grading_confidence=0.0,
            human_review_required=True,
            status="failed",
        )
        self.assertEqual(evaluate_grading_confidence(failed), 0.0)
        reviews = create_grading_reviews("eval_123", [failed])
        self.assertEqual(len(reviews), 1)
        self.assertIsNone(reviews[0].vlm_marks)
        self.assertEqual(reviews[0].reviewer_action, "pending")

    def test_creates_review_and_applies_override(self):
        res = EvaluationResult(
            question_id="Q1",
            marks_awarded=2.0,
            max_marks=5.0,
            reasoning="Partial answer",
            grading_confidence=0.5,
            human_review_required=True,
        )
        reviews = create_grading_reviews("eval_123", [res], confidence_threshold=0.7)
        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0].question_id, "Q1")
        self.assertEqual(reviews[0].reviewer_action, "pending")

        override = GradingReview(
            review_id="rev_1",
            evaluation_id="eval_123",
            question_id="Q1",
            vlm_marks=2.0,
            vlm_reasoning="Partial answer",
            human_marks=4.5,
            reviewer_action="overridden",
        )
        applied = apply_grading_override([res], override)
        self.assertTrue(applied)
        self.assertEqual(res.marks_awarded, 4.5)
        self.assertEqual(res.grading_confidence, 1.0)
        self.assertFalse(res.human_review_required)
        self.assertIn("Human Override: 4.5/5.0", res.reasoning)


if __name__ == "__main__":
    unittest.main()

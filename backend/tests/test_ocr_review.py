"""Unit tests for ocr_review."""

import unittest

from backend.app.models import OCRRegion, AnswerRegion, OCRReview, StudentAnswer
from backend.app.ocr_review import (
    apply_ocr_correction,
    create_ocr_reviews,
    evaluate_ocr_confidence,
)


class OCRReviewTests(unittest.TestCase):
    def test_low_confidence_for_garbage_text(self):
        conf = evaluate_ocr_confidence("###$$$%%%^^^&&&***")
        self.assertLess(conf, 0.7)

    def test_high_confidence_for_clean_english_text(self):
        conf = evaluate_ocr_confidence("Photosynthesis converts light energy into chemical energy stored in glucose.")
        self.assertGreaterEqual(conf, 0.8)

    def test_creates_review_items_for_low_confidence(self):
        bad_ans = StudentAnswer(
            question_id="Q1",
            text="$$$",
            ocr_confidence=0.2,
        )
        good_ans = StudentAnswer(
            question_id="Q2",
            text="Clean answer with sufficient details.",
            ocr_confidence=0.9,
        )
        reviews = create_ocr_reviews("eval_123", [bad_ans, good_ans], confidence_threshold=0.7)
        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0].question_id, "Q1")
        self.assertEqual(reviews[0].reviewer_action, "pending")

    def test_apply_ocr_correction_updates_answer(self):
        ans = StudentAnswer(question_id="Q1", text="Wrong text", ocr_confidence=0.3)
        review = OCRReview(
            review_id="rev_1",
            evaluation_id="eval_1",
            question_id="Q1",
            region_id="R1",
            original_text="Wrong text",
            corrected_text="Corrected human text",
            reviewer_action="corrected",
        )
        applied = apply_ocr_correction([ans], review)
        self.assertTrue(applied)
        self.assertEqual(ans.text, "Corrected human text")
        self.assertEqual(ans.ocr_confidence, 1.0)


if __name__ == "__main__":
    unittest.main()

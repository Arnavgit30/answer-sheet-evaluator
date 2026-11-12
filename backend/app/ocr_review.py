"""OCR confidence scoring and human review workflow."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any

from backend.app.models import OCRReview, StudentAnswer


def evaluate_ocr_confidence(text: str) -> float:
    """Calculate a heuristic confidence score (0.0 to 1.0) for OCR text.

    Considers:
    - Empty or near-empty text: low confidence
    - Character sanity: ratio of printable alphanumeric/punctuation to symbols
    - Excessive repetition of words (looping artifact)
    """
    clean = text.strip()
    if not clean:
        return 0.0
    if len(clean) < 10:
        return 0.4

    # 1. Printable ratio
    valid = sum(1 for c in clean if c.isalnum() or c.isspace() or c in ".,:;()[]-+=*/'\"%_!?")
    char_ratio = valid / len(clean)

    # 2. Check repetition (llama looping bug)
    words = clean.split()
    unique_ratio = len(set(words)) / max(len(words), 1)
    # If 50+ words but only 5 unique words, confidence is crushed
    rep_penalty = 0.5 if len(words) > 20 and unique_ratio < 0.2 else 0.0

    score = (char_ratio * 0.7) + (min(1.0, unique_ratio) * 0.3) - rep_penalty
    return round(max(0.0, min(1.0, score)), 2)


def create_ocr_reviews(
    evaluation_id: str,
    answers: list[StudentAnswer],
    confidence_threshold: float = 0.7,
) -> list[OCRReview]:
    """Identify low-confidence answers and create pending review items."""
    reviews: list[OCRReview] = []
    for ans in answers:
        score = ans.ocr_confidence if ans.ocr_confidence is not None else evaluate_ocr_confidence(ans.text)
        if score < confidence_threshold:
            first_reg_id = ans.source_regions[0].region.region_id if ans.source_regions else "R1"
            first_img = ans.answer_images[0] if ans.answer_images else ""
            reviews.append(
                OCRReview(
                    review_id=f"ocr_rev_{evaluation_id}_{ans.question_id}",
                    evaluation_id=evaluation_id,
                    question_id=ans.question_id,
                    region_id=first_reg_id,
                    original_text=ans.text,
                    original_image_path=first_img,
                    reviewer_action="pending",
                )
            )
    return reviews


def apply_ocr_correction(
    answers: list[StudentAnswer],
    review: OCRReview,
) -> bool:
    """Apply approved or edited text correction to the matching student answer."""
    for ans in answers:
        if ans.question_id == review.question_id:
            if review.reviewer_action == "corrected" and review.corrected_text is not None:
                ans.text = review.corrected_text
                ans.ocr_confidence = 1.0  # human verified
                return True
            elif review.reviewer_action == "confirmed":
                ans.ocr_confidence = 1.0
                return True
    return False

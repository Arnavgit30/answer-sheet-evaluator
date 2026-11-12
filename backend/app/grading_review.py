"""Grading confidence assessment and human review workflow."""

from __future__ import annotations

from backend.app.models import EvaluationResult, GradingReview


def evaluate_grading_confidence(
    result: EvaluationResult,
    baseline_similarity: float | None = None,
) -> float:
    """Calculate confidence in a grading decision (0.0 to 1.0).

    Flags cases where the evaluation failed/is pending, or where reasoning is
    missing or unusually brief. Text similarity divergence does not penalize
    confidence so valid alternative solutions are not penalized.
    """
    if result.marks_awarded is None or result.status in {"failed", "pending"}:
        return 0.0
    if result.max_marks <= 0:
        return 1.0

    reasoning_words = len((result.reasoning or "").split())
    if reasoning_words < 5:
        base = 0.4
    elif reasoning_words < 15:
        base = 0.7
    else:
        base = 0.9

    if result.grading_confidence is not None:
        llm_conf = float(result.grading_confidence)
        if reasoning_words < 5:
            base = min(base, llm_conf)
        else:
            base = llm_conf

    return round(max(0.0, min(1.0, base)), 2)


def create_grading_reviews(
    evaluation_id: str,
    results: list[EvaluationResult],
    confidence_threshold: float = 0.7,
) -> list[GradingReview]:
    """Identify failed, pending, or low-confidence grading decisions and create pending review items."""
    reviews: list[GradingReview] = []
    for r in results:
        conf = evaluate_grading_confidence(r)
        needs_review = (
            bool(r.human_review_required)
            or r.marks_awarded is None
            or r.status in {"failed", "pending"}
            or conf < confidence_threshold
        )
        if needs_review:
            reviews.append(
                GradingReview(
                    review_id=f"grade_rev_{evaluation_id}_{r.question_id}",
                    evaluation_id=evaluation_id,
                    question_id=r.question_id,
                    vlm_marks=r.marks_awarded,
                    vlm_reasoning=r.reasoning,
                    human_marks=None,
                    reviewer_action="pending",
                )
            )
    return reviews


def apply_grading_override(
    results: list[EvaluationResult],
    review: GradingReview,
) -> bool:
    """Apply human grade override or confirmation to evaluation results."""
    for r in results:
        if r.question_id == review.question_id:
            if review.reviewer_action == "overridden" and review.human_marks is not None:
                r.marks_awarded = float(review.human_marks)
                r.grading_confidence = 1.0
                r.human_review_required = False
                r.status = "completed"
                r.reasoning = f"{r.reasoning}\n[Human Override: {review.human_marks}/{r.max_marks}]"
                return True
            elif review.reviewer_action == "confirmed":
                r.grading_confidence = 1.0
                r.human_review_required = False
                r.status = "completed"
                return True
    return False

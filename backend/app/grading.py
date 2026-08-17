"""Transparent, replaceable baseline grading using RapidFuzz text similarity."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class GradingResult:
    document_id: str
    grading_path: str
    question_count: int
    cached: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaselineSimilarityGrader:
    """A minimal interface that can later be replaced by another evaluator."""

    grading_method = "baseline_similarity"

    def grade(self, student_questions: list[dict[str, Any]], key_questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        students = {question.get("question_id"): question for question in student_questions if question.get("question_id")}
        keys = {question.get("question_id"): question for question in key_questions if question.get("question_id")}
        results = [self._grade_question(question_id, students.get(question_id), key) for question_id, key in keys.items()]
        results.extend(self._grade_question(question_id, student, None) for question_id, student in students.items() if question_id not in keys)
        return results

    def _grade_question(
        self, question_id: str, student: dict[str, Any] | None, key: dict[str, Any] | None
    ) -> dict[str, Any]:
        student_answer = student.get("answer_text", "") if student else ""
        reference_answer = key.get("answer_text", "") if key else ""
        similarity = _similarity(student_answer, reference_answer)
        maximum_marks = key.get("maximum_marks") if key else None
        predicted_marks = _predicted_marks(similarity, maximum_marks)
        return {
            "question_id": question_id,
            "student_answer": student_answer,
            "reference_answer": reference_answer,
            "similarity_score": similarity,
            "predicted_marks": predicted_marks,
            "maximum_marks": maximum_marks,
            "grading_method": self.grading_method,
        }


def grade_from_files(
    student_segmentation_path: str | Path,
    answer_key_path: str | Path,
    output_root: str | Path,
    grader: BaselineSimilarityGrader | None = None,
) -> GradingResult:
    """Grade one segmented student document and cache the complete baseline result."""
    student_path = Path(student_segmentation_path)
    key_path = Path(answer_key_path)
    student_bytes = student_path.read_bytes()
    key_bytes = key_path.read_bytes()
    student_sha256 = hashlib.sha256(student_bytes).hexdigest()
    key_sha256 = hashlib.sha256(key_bytes).hexdigest()
    student_data = json.loads(student_bytes.decode("utf-8"))
    key_data = json.loads(key_bytes.decode("utf-8"))
    document_id = str(student_data["document_id"])
    target = Path(output_root) / _safe_document_id(document_id) / "grading.json"
    if _is_current_cache(target, document_id, student_sha256, key_sha256):
        payload = json.loads(target.read_text(encoding="utf-8"))
        return GradingResult(document_id, str(target.resolve()), len(payload["questions"]), cached=True)

    results = (grader or BaselineSimilarityGrader()).grade(student_data.get("questions", []), key_data.get("questions", []))
    total_predicted, total_maximum = _totals(results)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "document_id": document_id,
                "student_segmentation_sha256": student_sha256,
                "answer_key_sha256": key_sha256,
                "grading_method": BaselineSimilarityGrader.grading_method,
                "questions": results,
                "total_predicted_marks": total_predicted,
                "total_maximum_marks": total_maximum,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return GradingResult(document_id, str(target.resolve()), len(results), cached=False)


def normalize_text(text: object) -> str:
    """Lowercase and normalize whitespace/punctuation before baseline matching."""
    value = unicodedata.normalize("NFKC", str(text or "")).lower()
    value = re.sub(r"[^\w\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _similarity(student_answer: object, reference_answer: object) -> float:
    student = normalize_text(student_answer)
    reference = normalize_text(reference_answer)
    if not student or not reference:
        return 0.0
    from rapidfuzz import fuzz

    return float(fuzz.ratio(student, reference))


def _predicted_marks(similarity: float, maximum_marks: object) -> float | None:
    if isinstance(maximum_marks, bool) or not isinstance(maximum_marks, (int, float)):
        return None
    return max(0.0, min(float(maximum_marks), round(float(maximum_marks) * similarity / 100.0, 2)))


def _totals(results: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    if any(result["maximum_marks"] is None or result["predicted_marks"] is None for result in results):
        return None, None
    return (
        round(sum(float(result["predicted_marks"]) for result in results), 2),
        round(sum(float(result["maximum_marks"]) for result in results), 2),
    )


def _is_current_cache(path: Path, document_id: str, student_sha256: str, key_sha256: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return (
            payload["document_id"] == document_id
            and payload["student_segmentation_sha256"] == student_sha256
            and payload["answer_key_sha256"] == key_sha256
        )
    except (KeyError, TypeError, json.JSONDecodeError):
        return False


def _safe_document_id(document_id: str) -> str:
    if not document_id or Path(document_id).name != document_id:
        raise ValueError(f"Invalid document_id: {document_id}")
    return document_id

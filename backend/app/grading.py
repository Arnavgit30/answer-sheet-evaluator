"""Primary LLM/VLM grading with optional passive text-similarity metric for benchmarking."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from backend.app.models import (
    Chart,
    Diagram,
    Formula,
    ReferenceAnswer,
    StudentAnswer,
    Table,
)
from backend.app.vlm_evaluator import VLMEvaluator


@dataclass
class GradingResult:
    document_id: str
    grading_path: str
    question_count: int
    cached: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLMGrader:
    """Grades student answers using VLMEvaluator (LLM) as the primary grader.

    Text similarity is computed only as passive metadata and never influences marks.
    """

    def __init__(self, evaluator: VLMEvaluator | None = None) -> None:
        self.evaluator = evaluator or VLMEvaluator()
        self.grading_method = self.evaluator.model

    def grade(
        self, student_questions: list[dict[str, Any]], key_questions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        students = {
            q.get("question_id"): q for q in student_questions if q.get("question_id")
        }
        keys = {
            q.get("question_id"): q for q in key_questions if q.get("question_id")
        }
        results = [
            self._grade_question(qid, students.get(qid), key)
            for qid, key in keys.items()
        ]
        results.extend(
            self._grade_question(qid, student, None)
            for qid, student in students.items()
            if qid not in keys
        )
        return results

    def _grade_question(
        self,
        question_id: str,
        student: dict[str, Any] | None,
        key: dict[str, Any] | None,
    ) -> dict[str, Any]:
        student_answer = (student.get("answer_text") or student.get("text") or "") if student else ""
        reference_answer = (key.get("answer_text") or "") if key else ""
        similarity = _semantic_similarity(student_answer, reference_answer)
        maximum_marks = key.get("maximum_marks") if key else None

        student_obj = _to_student_answer(question_id, student)
        ref_obj = _to_reference_answer(question_id, key)

        eval_result = self.evaluator.evaluate_question(
            student_obj,
            ref_obj,
            baseline_similarity=similarity,
        )

        return {
            "question_id": question_id,
            "student_answer": student_answer,
            "reference_answer": reference_answer,
            "similarity_score": similarity,
            "predicted_marks": eval_result.marks_awarded,
            "maximum_marks": maximum_marks if maximum_marks is not None else eval_result.max_marks,
            "grading_method": eval_result.grading_method,
            "reasoning": eval_result.reasoning,
            "concepts_present": eval_result.concepts_present,
            "concepts_missing": eval_result.concepts_missing,
            "errors": eval_result.errors,
            "grading_confidence": eval_result.grading_confidence,
            "human_review_required": eval_result.human_review_required,
            "status": eval_result.status,
        }


# Backward-compatible aliases so existing imports continue to work.
SemanticSimilarityGrader = LLMGrader
BaselineSimilarityGrader = LLMGrader


def grade_from_files(
    student_segmentation_path: str | Path,
    answer_key_path: str | Path,
    output_root: str | Path,
    grader: LLMGrader | None = None,
) -> GradingResult:
    """Grade one segmented student document using the LLM grader and cache completed results."""
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

    active_grader = grader or LLMGrader()
    results = active_grader.grade(
        student_data.get("questions", []), key_data.get("questions", [])
    )
    total_predicted, total_maximum = _totals(results)
    methods = {r.get("grading_method") for r in results if r.get("grading_method")}
    overall_method = active_grader.grading_method if len(methods) != 1 else next(iter(methods))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "document_id": document_id,
                "student_segmentation_sha256": student_sha256,
                "answer_key_sha256": key_sha256,
                "grading_method": overall_method,
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
    """Lowercase and normalize whitespace/punctuation before similarity matching."""
    value = unicodedata.normalize("NFKC", str(text or "")).lower()
    value = re.sub(r"[^\w\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _semantic_similarity(student_answer: object, reference_answer: object) -> float:
    """Return 0-100 passive similarity score for optional benchmarking only."""
    student = normalize_text(student_answer)
    reference = normalize_text(reference_answer)
    if not student or not reference:
        return 0.0
    try:
        from rapidfuzz import fuzz
        return round(float(fuzz.ratio(student, reference)), 2)
    except ImportError:
        s_words, r_words = set(student.split()), set(reference.split())
        return round(len(s_words & r_words) / max(len(s_words | r_words), 1) * 100.0, 2)


def _to_student_answer(question_id: str, student: dict[str, Any] | None) -> StudentAnswer:
    if not student:
        return StudentAnswer(question_id=question_id, text="")
    text = str(student.get("answer_text") or student.get("text") or "")
    pages = list(student.get("source_pages") or student.get("pages") or [])
    answer_images = list(student.get("answer_images") or [])
    formulas = [
        f if isinstance(f, Formula) else Formula(
            formula_id=str(f.get("formula_id", f"{question_id}_F{i+1}")),
            raw_text=str(f.get("raw_text", "")),
            source_region_id=str(f.get("source_region_id", "")),
            page_number=int(f.get("page_number", 1)),
            bbox=f.get("bbox"),
        )
        for i, f in enumerate(student.get("formulas") or [])
        if isinstance(f, (Formula, dict))
    ]
    tables = [
        t if isinstance(t, Table) else Table(
            table_id=str(t.get("table_id", f"{question_id}_T{i+1}")),
            raw_text=str(t.get("raw_text", "")),
            source_region_id=str(t.get("source_region_id", "")),
            page_number=int(t.get("page_number", 1)),
            bbox=t.get("bbox"),
        )
        for i, t in enumerate(student.get("tables") or [])
        if isinstance(t, (Table, dict))
    ]
    charts = [
        c if isinstance(c, Chart) else Chart(
            chart_id=str(c.get("chart_id", f"{question_id}_C{i+1}")),
            description=str(c.get("description", "")),
            source_region_id=str(c.get("source_region_id", "")),
            page_number=int(c.get("page_number", 1)),
            bbox=c.get("bbox"),
        )
        for i, c in enumerate(student.get("charts") or [])
        if isinstance(c, (Chart, dict))
    ]
    diagrams = [
        d if isinstance(d, Diagram) else Diagram(
            diagram_id=str(d.get("diagram_id", f"{question_id}_D{i+1}")),
            description=str(d.get("description", "")),
            source_region_id=str(d.get("source_region_id", "")),
            page_number=int(d.get("page_number", 1)),
            bbox=d.get("bbox"),
        )
        for i, d in enumerate(student.get("diagrams") or [])
        if isinstance(d, (Diagram, dict))
    ]
    return StudentAnswer(
        question_id=question_id,
        pages=pages,
        answer_images=answer_images,
        text=text,
        formulas=formulas,
        tables=tables,
        charts=charts,
        diagrams=diagrams,
        ocr_confidence=student.get("ocr_confidence"),
    )


def _to_reference_answer(question_id: str, key: dict[str, Any] | None) -> ReferenceAnswer:
    if not key:
        return ReferenceAnswer(question_id=question_id, answer_text="", maximum_marks=5.0)
    return ReferenceAnswer(
        question_id=question_id,
        answer_text=str(key.get("answer_text") or ""),
        maximum_marks=key.get("maximum_marks") if key.get("maximum_marks") is not None else 5.0,
        source_pages=list(key.get("source_pages") or []),
        rubric=key.get("rubric") or key.get("marking_scheme"),
        key_concepts=list(key.get("key_concepts") or []),
    )


def _totals(results: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    if any(result["maximum_marks"] is None or result["predicted_marks"] is None for result in results):
        total_max = (
            round(sum(float(r["maximum_marks"]) for r in results), 2)
            if results and all(r["maximum_marks"] is not None for r in results)
            else None
        )
        return None, total_max
    return (
        round(sum(float(result["predicted_marks"]) for result in results), 2),
        round(sum(float(result["maximum_marks"]) for result in results), 2),
    )


def _is_current_cache(path: Path, document_id: str, student_sha256: str, key_sha256: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("grading_method") in {"baseline_similarity", "baseline_fallback", "llm_pending", "llm_failed"}:
            return False
        questions = payload.get("questions", [])
        if any(q.get("status") in {"pending", "failed"} or q.get("grading_method") == "baseline_similarity" for q in questions):
            return False
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

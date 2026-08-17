"""Orchestration layer between API routes and existing pipeline stages."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app.answer_key_parser import parse_answer_key_file, process_answer_key_document
from backend.app.benchmarking import benchmark_grading
from backend.app.dataset_adapter import build_manifest, validate_dataset
from backend.app.evaluation_store import EvaluationStore
from backend.app.grading import grade_from_files
from backend.app.normalization import normalize_raw_ocr
from backend.app.ocr_service import LlamaCppOCRService
from backend.app.preprocessing import preprocess_document
from backend.app.segmentation import segment_normalized_document


class EvaluationService:
    def __init__(self, project_root: str | Path | None = None) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
        self.outputs = self.project_root / "outputs"
        self.store = EvaluationStore(self.outputs / "evaluations")

    def validate_dataset(self, config_path: str) -> dict[str, Any]:
        result = validate_dataset(self._config_path(config_path))
        result["errors"] = [self._hide_path(error) for error in result["errors"]]
        if result["manifest"]:
            result["manifest"] = _public_manifest(result["manifest"])
        return result

    def start_evaluation(self, config_path: str, student_document_id: str | None = None) -> dict[str, Any]:
        try:
            manifest = build_manifest(self._config_path(config_path))
        except ValueError as exc:
            raise ValueError(self._hide_path(str(exc))) from exc
        student = _select_document(manifest.student_documents, student_document_id)
        answer_key = next((item for item in manifest.answer_keys if item.document_id == student.answer_key_id), None)
        if answer_key is None:
            raise ValueError(f"Answer key {student.answer_key_id} is not available")
        evaluation = {
            "evaluation_id": uuid.uuid4().hex,
            "status": "queued",
            "dataset_id": manifest.dataset_id,
            "document_id": student.document_id,
            "answer_key_id": answer_key.document_id,
            "created_at": _now(),
            "updated_at": _now(),
            "error": None,
            "result": None,
        }
        self.store.save(evaluation)
        evaluation["status"] = "running"
        evaluation["updated_at"] = _now()
        self.store.save(evaluation)
        try:
            processed = preprocess_document(student, self.outputs / "processed")
            raw_ocr = LlamaCppOCRService().ocr_document(processed, self.outputs / "ocr")
            normalized = normalize_raw_ocr(raw_ocr.raw_ocr_path, self.outputs / "normalized")
            segmented = segment_normalized_document(normalized.normalized_path, self.outputs / "segmentation")
            parsed_key = self._process_answer_key(answer_key)
            grading = grade_from_files(segmented.segmentation_path, parsed_key, self.outputs / "grading")
            benchmark = benchmark_grading(grading.grading_path, manifest.human_marks_path, self.outputs / "benchmark")
            evaluation["result"] = _public_result(
                grading.grading_path, benchmark.benchmark_path, segmented.segmentation_path
            )
            evaluation["status"] = "completed"
        except Exception as exc:
            evaluation["status"] = "failed"
            evaluation["error"] = self._hide_path(str(exc))
        evaluation["updated_at"] = _now()
        self.store.save(evaluation)
        return _public_evaluation(evaluation)

    def get_evaluation(self, evaluation_id: str) -> dict[str, Any] | None:
        record = self.store.get(evaluation_id)
        return _public_evaluation(record) if record else None

    def get_results(self, evaluation_id: str) -> dict[str, Any] | None:
        record = self.store.get(evaluation_id)
        if not record or record["status"] != "completed":
            return None
        return {"evaluation_id": record["evaluation_id"], "status": record["status"], **record["result"]}

    def _process_answer_key(self, answer_key: Any) -> str:
        suffixes = {Path(file).suffix.lower() for file in answer_key.files}
        if suffixes <= {".pdf", ".png", ".jpg", ".jpeg"}:
            return process_answer_key_document(answer_key, self.outputs / "answer_keys").answer_key_path
        if len(answer_key.files) == 1 and suffixes <= {".txt", ".json"}:
            return parse_answer_key_file(
                answer_key.files[0], self.outputs / "answer_keys" / "parsed", answer_key_id=answer_key.document_id
            ).answer_key_path
        raise ValueError("Answer key must be one PDF/image document or one text/JSON file")

    def _config_path(self, value: str) -> Path:
        path = Path(value)
        resolved = (path if path.is_absolute() else self.project_root / path).resolve()
        try:
            resolved.relative_to(self.project_root)
        except ValueError as exc:
            raise ValueError("config_path must be inside this demo project") from exc
        return resolved

    def _hide_path(self, message: str) -> str:
        return message.replace(str(self.project_root), ".")


def _select_document(documents: list[Any], requested_id: str | None) -> Any:
    if not documents:
        raise ValueError("Dataset contains no student documents")
    if requested_id is None:
        return documents[0]
    for document in documents:
        if document.document_id == requested_id:
            return document
    raise ValueError(f"Student document not found: {requested_id}")


def _public_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": manifest["dataset_id"],
        "student_documents": [
            {
                "document_id": document["document_id"],
                "page_or_file_count": len(document["files"]),
                "answer_key_id": document["answer_key_id"],
                "human_marks_available": document["human_marks_available"],
            }
            for document in manifest["student_documents"]
        ],
        "answer_keys": [
            {"document_id": document["document_id"], "page_or_file_count": len(document["files"])}
            for document in manifest["answer_keys"]
        ],
        "human_marks_available": bool(manifest["human_marks_path"]),
    }


def _public_evaluation(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record[key] for key in ("evaluation_id", "status", "dataset_id", "document_id", "answer_key_id", "created_at", "updated_at", "error")}


def _public_result(grading_path: str, benchmark_path: str, segmentation_path: str) -> dict[str, Any]:
    grading = json.loads(Path(grading_path).read_text(encoding="utf-8"))
    benchmark = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    segmentation = json.loads(Path(segmentation_path).read_text(encoding="utf-8"))
    figures = {question["question_id"]: question.get("figures", []) for question in segmentation["questions"]}
    questions = []
    for question in grading["questions"]:
        public_question = dict(question)
        public_question["figures"] = figures.get(question["question_id"], [])
        questions.append(public_question)
    return {
        "grading": {
            "grading_method": grading["grading_method"],
            "questions": questions,
            "total_predicted_marks": grading["total_predicted_marks"],
            "total_maximum_marks": grading["total_maximum_marks"],
        },
        "benchmark": benchmark["metrics"],
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

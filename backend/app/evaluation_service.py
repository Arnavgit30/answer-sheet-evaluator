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

            # Structured representation & reviews
            import hashlib
            from dataclasses import asdict
            from backend.app.answer_builder import build_student_answers
            from backend.app.grading import _semantic_similarity, _totals
            from backend.app.grading_review import create_grading_reviews
            from backend.app.models import ReferenceAnswer, StudentAnswer
            from backend.app.ocr_review import create_ocr_reviews
            from backend.app.vlm_evaluator import VLMEvaluator

            norm_data = json.loads(Path(normalized.normalized_path).read_text(encoding="utf-8"))
            student_answers = build_student_answers(norm_data.get("pages", []))
            page_files = {p.page_number: p.processed_file for p in processed.pages}
            for a in student_answers:
                a.answer_images = [page_files[pn] for pn in a.pages if pn in page_files]

            ocr_reviews = create_ocr_reviews(evaluation["evaluation_id"], student_answers)

            # VLM / LLM Primary Grading
            key_path_obj = Path(parsed_key)
            key_bytes = key_path_obj.read_bytes()
            key_data = json.loads(key_bytes.decode("utf-8"))
            ref_answers = {
                q["question_id"]: ReferenceAnswer(
                    question_id=q["question_id"],
                    answer_text=q.get("answer_text", ""),
                    maximum_marks=q.get("maximum_marks") if q.get("maximum_marks") is not None else 5.0,
                    source_pages=q.get("source_pages", []),
                    rubric=q.get("rubric") or q.get("marking_scheme"),
                    key_concepts=q.get("key_concepts", []),
                )
                for q in key_data.get("questions", [])
            }

            student_map = {a.question_id: a for a in student_answers}
            ordered_qids = list(ref_answers.keys()) + [
                a.question_id for a in student_answers if a.question_id not in ref_answers
            ]
            for qid in ordered_qids:
                if qid not in student_map:
                    blank_ans = StudentAnswer(question_id=qid, text="")
                    student_answers.append(blank_ans)
                    student_map[qid] = blank_ans

            evaluator = VLMEvaluator()
            vlm_results = []
            grading_questions = []
            for qid in ordered_qids:
                ans = student_map[qid]
                ref = ref_answers.get(qid, ReferenceAnswer(qid, "", 5.0))
                similarity = _semantic_similarity(ans.text, ref.answer_text)
                vlm_res = evaluator.evaluate_question(
                    ans,
                    ref,
                    baseline_similarity=similarity,
                )
                vlm_results.append(vlm_res)
                grading_questions.append(
                    {
                        "question_id": qid,
                        "student_answer": ans.text,
                        "reference_answer": ref.answer_text,
                        "similarity_score": similarity,
                        "predicted_marks": vlm_res.marks_awarded,
                        "maximum_marks": ref.maximum_marks,
                        "grading_method": vlm_res.grading_method,
                        "reasoning": vlm_res.reasoning,
                        "concepts_present": vlm_res.concepts_present,
                        "concepts_missing": vlm_res.concepts_missing,
                        "errors": vlm_res.errors,
                        "grading_confidence": vlm_res.grading_confidence,
                        "human_review_required": vlm_res.human_review_required,
                        "status": vlm_res.status,
                    }
                )

            seg_bytes = Path(segmented.segmentation_path).read_bytes()
            total_predicted, total_maximum = _totals(grading_questions)
            grading_target = self.outputs / "grading" / student.document_id / "grading.json"
            grading_target.parent.mkdir(parents=True, exist_ok=True)
            grading_target.write_text(
                json.dumps(
                    {
                        "document_id": student.document_id,
                        "student_segmentation_sha256": hashlib.sha256(seg_bytes).hexdigest(),
                        "answer_key_sha256": hashlib.sha256(key_bytes).hexdigest(),
                        "grading_method": evaluator.model,
                        "questions": grading_questions,
                        "total_predicted_marks": total_predicted,
                        "total_maximum_marks": total_maximum,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            benchmark = benchmark_grading(
                grading_target, manifest.human_marks_path, self.outputs / "benchmark"
            )

            grading_reviews = create_grading_reviews(evaluation["evaluation_id"], vlm_results)

            evaluation["student_answers"] = [a.to_dict() for a in student_answers]
            evaluation["reference_answers"] = [r.to_dict() for r in ref_answers.values()]
            evaluation["vlm_grading"] = [r.to_dict() for r in vlm_results]
            evaluation["ocr_reviews"] = [r.to_dict() for r in ocr_reviews]
            evaluation["grading_reviews"] = [r.to_dict() for r in grading_reviews]
            evaluation["processed_pages"] = [asdict(p) for p in processed.pages]

            evaluation["result"] = _public_result(
                str(grading_target), benchmark.benchmark_path, segmented.segmentation_path
            )
            evaluation["result"]["vlm_grading"] = [r.to_dict() for r in vlm_results]
            evaluation["result"]["ocr_reviews_count"] = len(ocr_reviews)
            evaluation["result"]["grading_reviews_count"] = len(grading_reviews)
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

    def get_questions(self, evaluation_id: str) -> list[dict[str, Any]]:
        record = self.store.get(evaluation_id)
        if not record:
            return []
        answers = {a["question_id"]: a for a in record.get("student_answers", [])}
        refs = {r["question_id"]: r for r in record.get("reference_answers", [])}
        vlm = {v["question_id"]: v for v in record.get("vlm_grading", [])}
        baseline = {q["question_id"]: q for q in record.get("result", {}).get("grading", {}).get("questions", [])}
        all_qids = sorted(set(answers) | set(refs) | set(vlm))
        return [
            {
                "question_id": qid,
                "student_answer": answers.get(qid),
                "reference_answer": refs.get(qid),
                "vlm_evaluation": vlm.get(qid),
                "baseline_evaluation": baseline.get(qid),
            }
            for qid in all_qids
        ]

    def get_question_detail(self, evaluation_id: str, question_id: str) -> dict[str, Any] | None:
        questions = self.get_questions(evaluation_id)
        return next((q for q in questions if q["question_id"] == question_id), None)

    def get_ocr_reviews(self, evaluation_id: str) -> list[dict[str, Any]]:
        record = self.store.get(evaluation_id)
        return record.get("ocr_reviews", []) if record else []

    def update_ocr_review(
        self,
        evaluation_id: str,
        review_id: str,
        action: str,
        corrected_text: str | None = None,
    ) -> dict[str, Any] | None:
        record = self.store.get(evaluation_id)
        if not record:
            return None
        reviews = record.get("ocr_reviews", [])
        target = next((r for r in reviews if r["review_id"] == review_id), None)
        if not target:
            return None
        target["reviewer_action"] = action
        if corrected_text is not None:
            target["corrected_text"] = corrected_text
        target["reviewed_at"] = _now()
        if action == "corrected" and corrected_text is not None:
            for ans in record.get("student_answers", []):
                if ans["question_id"] == target["question_id"]:
                    ans["text"] = corrected_text
                    ans["ocr_confidence"] = 1.0
        self.store.save(record)
        return target

    def get_grading_reviews(self, evaluation_id: str) -> list[dict[str, Any]]:
        record = self.store.get(evaluation_id)
        return record.get("grading_reviews", []) if record else []

    def update_grading_review(
        self,
        evaluation_id: str,
        review_id: str,
        action: str,
        human_marks: float | None = None,
        notes: str | None = None,
    ) -> dict[str, Any] | None:
        record = self.store.get(evaluation_id)
        if not record:
            return None
        reviews = record.get("grading_reviews", [])
        target = next((r for r in reviews if r["review_id"] == review_id), None)
        if not target:
            return None
        target["reviewer_action"] = action
        if human_marks is not None:
            target["human_marks"] = human_marks
        target["reviewed_at"] = _now()
        if action == "overridden" and human_marks is not None:
            for gr in record.get("vlm_grading", []):
                if gr["question_id"] == target["question_id"]:
                    gr["marks_awarded"] = human_marks
                    gr["grading_confidence"] = 1.0
                    gr["human_review_required"] = False
                    gr["status"] = "completed"
                    if notes:
                        gr["reasoning"] += f"\n[Override note: {notes}]"
            grading_block = record.get("result", {}).get("grading", {})
            for q in grading_block.get("questions", []):
                if q["question_id"] == target["question_id"]:
                    q["predicted_marks"] = human_marks
                    q["grading_confidence"] = 1.0
                    q["human_review_required"] = False
                    q["status"] = "completed"
            if grading_block.get("questions"):
                from backend.app.grading import _totals
                total_pred, total_max = _totals(grading_block["questions"])
                grading_block["total_predicted_marks"] = total_pred
                grading_block["total_maximum_marks"] = total_max
        elif action == "confirmed":
            for gr in record.get("vlm_grading", []):
                if gr["question_id"] == target["question_id"]:
                    gr["grading_confidence"] = 1.0
                    gr["human_review_required"] = False
                    gr["status"] = "completed"
            for q in record.get("result", {}).get("grading", {}).get("questions", []):
                if q["question_id"] == target["question_id"]:
                    q["grading_confidence"] = 1.0
                    q["human_review_required"] = False
                    q["status"] = "completed"
        self.store.save(record)
        return target

    def get_page_image_path(self, evaluation_id: str, page_number: int) -> Path | None:
        record = self.store.get(evaluation_id)
        if not record:
            return None
        for p in record.get("processed_pages", []):
            if p.get("page_number") == page_number:
                path = Path(p["processed_file"])
                if path.is_file():
                    return path
        return None

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

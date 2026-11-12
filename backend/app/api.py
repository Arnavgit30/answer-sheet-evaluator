"""HTTP routes only; pipeline orchestration lives in EvaluationService."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.api_models import (
    DatasetValidationRequest,
    DatasetValidationResponse,
    EvaluationRequest,
    EvaluationResponse,
    EvaluationResultsResponse,
    GradingReviewActionRequest,
    HealthResponse,
    OCRReviewActionRequest,
)
from backend.app.evaluation_service import EvaluationService


def create_router(service: EvaluationService) -> APIRouter:
    router = APIRouter(prefix="/api/v1")

    @router.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @router.post("/datasets/validate", response_model=DatasetValidationResponse)
    def validate_dataset(request: DatasetValidationRequest) -> dict:
        try:
            return service.validate_dataset(request.config_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/evaluations", response_model=EvaluationResponse)
    def create_evaluation(request: EvaluationRequest) -> dict:
        try:
            return service.start_evaluation(request.config_path, request.student_document_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/evaluations/{evaluation_id}", response_model=EvaluationResponse)
    def get_evaluation(evaluation_id: str) -> dict:
        evaluation = service.get_evaluation(evaluation_id)
        if not evaluation:
            raise HTTPException(status_code=404, detail="Evaluation not found")
        return evaluation

    @router.get("/evaluations/{evaluation_id}/results", response_model=EvaluationResultsResponse)
    def get_results(evaluation_id: str) -> dict:
        results = service.get_results(evaluation_id)
        if not results:
            raise HTTPException(status_code=404, detail="Completed evaluation results not found")
        return results

    @router.get("/evaluations/{evaluation_id}/questions")
    def get_questions(evaluation_id: str) -> list[dict]:
        return service.get_questions(evaluation_id)

    @router.get("/evaluations/{evaluation_id}/questions/{question_id}")
    def get_question_detail(evaluation_id: str, question_id: str) -> dict:
        q = service.get_question_detail(evaluation_id, question_id)
        if not q:
            raise HTTPException(status_code=404, detail="Question not found")
        return q

    @router.get("/evaluations/{evaluation_id}/pages/{page_number}")
    def get_page_image(evaluation_id: str, page_number: int):
        from fastapi.responses import FileResponse
        path = service.get_page_image_path(evaluation_id, page_number)
        if not path or not path.is_file():
            raise HTTPException(status_code=404, detail="Page image not found")
        media = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return FileResponse(path, media_type=media)

    @router.get("/evaluations/{evaluation_id}/ocr-reviews")
    def get_ocr_reviews(evaluation_id: str) -> list[dict]:
        return service.get_ocr_reviews(evaluation_id)

    @router.post("/evaluations/{evaluation_id}/ocr-reviews/{review_id}")
    def update_ocr_review(
        evaluation_id: str,
        review_id: str,
        body: OCRReviewActionRequest,
    ) -> dict:
        updated = service.update_ocr_review(
            evaluation_id,
            review_id,
            body.action,
            body.corrected_text,
        )
        if not updated:
            raise HTTPException(status_code=404, detail="OCR review item not found")
        return updated

    @router.get("/evaluations/{evaluation_id}/grading-reviews")
    def get_grading_reviews(evaluation_id: str) -> list[dict]:
        return service.get_grading_reviews(evaluation_id)

    @router.post("/evaluations/{evaluation_id}/grading-reviews/{review_id}")
    def update_grading_review(
        evaluation_id: str,
        review_id: str,
        body: GradingReviewActionRequest,
    ) -> dict:
        updated = service.update_grading_review(
            evaluation_id,
            review_id,
            body.action,
            body.human_marks,
            body.notes,
        )
        if not updated:
            raise HTTPException(status_code=404, detail="Grading review item not found")
        return updated

    return router

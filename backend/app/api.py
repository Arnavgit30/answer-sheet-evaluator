"""HTTP routes only; pipeline orchestration lives in EvaluationService."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.api_models import (
    DatasetValidationRequest,
    DatasetValidationResponse,
    EvaluationRequest,
    EvaluationResponse,
    EvaluationResultsResponse,
    HealthResponse,
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

    return router

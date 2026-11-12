"""Pydantic request and response models for the demo API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DatasetValidationRequest(BaseModel):
    config_path: str = "configs/dataset_config.yaml"


class DatasetValidationResponse(BaseModel):
    valid: bool
    errors: list[str]
    manifest: dict[str, Any] | None


class EvaluationRequest(BaseModel):
    config_path: str = "configs/dataset_config.yaml"
    student_document_id: str | None = None


class EvaluationResponse(BaseModel):
    evaluation_id: str
    status: Literal["queued", "running", "completed", "failed"]
    dataset_id: str
    document_id: str
    answer_key_id: str
    created_at: str
    updated_at: str
    error: str | None = None


class EvaluationResultsResponse(BaseModel):
    evaluation_id: str
    status: Literal["completed"]
    grading: dict[str, Any]
    benchmark: dict[str, Any]


class HealthResponse(BaseModel):
    status: str = Field(default="ok")


class OCRReviewActionRequest(BaseModel):
    action: Literal["confirmed", "corrected"]
    corrected_text: str | None = None
    notes: str | None = None


class GradingReviewActionRequest(BaseModel):
    action: Literal["confirmed", "overridden"]
    human_marks: float | None = None
    notes: str | None = None

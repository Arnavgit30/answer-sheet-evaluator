"""Central dataclass models for GradeP v2."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Page:
    page_number: int
    source_file: str
    processed_file: str
    width: int | None = None
    height: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OCRRegion:
    region_id: str
    page_number: int
    content_type: str  # "text" | "formula" | "table" | "chart" | "diagram"
    text: str | None = None
    raw_content: str | None = None
    bbox: list[float] | None = None
    polygon: list[list[float]] | None = None
    source_prompt: str = "OCR:"
    source_block_id: int | None = None
    order: int | None = None
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QuestionBoundary:
    question_id: str
    start_page: int
    start_region_id: str
    end_page: int | None = None
    end_region_id: str | None = None
    detection_method: str = "regex_heuristic"
    detection_confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnswerRegion:
    question_id: str
    region: OCRRegion
    is_boundary_region: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Formula:
    formula_id: str
    raw_text: str
    source_region_id: str
    page_number: int
    bbox: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Table:
    table_id: str
    raw_text: str
    source_region_id: str
    page_number: int
    bbox: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Chart:
    chart_id: str
    description: str
    source_region_id: str
    page_number: int
    bbox: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Diagram:
    diagram_id: str
    description: str
    source_region_id: str
    page_number: int
    bbox: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StudentAnswer:
    question_id: str
    pages: list[int] = field(default_factory=list)
    source_regions: list[AnswerRegion] = field(default_factory=list)
    answer_images: list[str] = field(default_factory=list)
    text: str = ""
    formulas: list[Formula] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    charts: list[Chart] = field(default_factory=list)
    diagrams: list[Diagram] = field(default_factory=list)
    ocr_confidence: float | None = None
    subparts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReferenceAnswer:
    question_id: str
    answer_text: str
    maximum_marks: float | None = None
    source_pages: list[int] = field(default_factory=list)
    rubric: str | None = None
    key_concepts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationResult:
    question_id: str
    marks_awarded: float | None
    max_marks: float
    reasoning: str
    concepts_present: list[str] = field(default_factory=list)
    concepts_missing: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    grading_confidence: float | None = None
    grading_method: str = "qwen/qwen3.8-27b"
    baseline_similarity: float | None = None
    baseline_marks: float | None = None
    human_review_required: bool = False
    status: str = "completed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OCRReview:
    review_id: str
    evaluation_id: str
    question_id: str
    region_id: str
    original_text: str
    corrected_text: str | None = None
    original_image_path: str = ""
    reviewer_action: str = "pending"  # pending | confirmed | corrected
    reviewed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GradingReview:
    review_id: str
    evaluation_id: str
    question_id: str
    vlm_marks: float | None
    vlm_reasoning: str
    human_marks: float | None = None
    reviewer_action: str = "pending"  # pending | confirmed | overridden
    reviewed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

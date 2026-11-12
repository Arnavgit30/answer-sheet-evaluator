"""Build structured StudentAnswer objects from associated answer regions."""

from __future__ import annotations

import re
from typing import Any

from backend.app.models import (
    AnswerRegion,
    Chart,
    Diagram,
    Formula,
    StudentAnswer,
    Table,
)
from backend.app.question_detector import DEFAULT_MARKER_PATTERN
from backend.app.region_associator import associate_regions

_SUBPART_RE = re.compile(r"\((?P<subpart>[A-Za-z0-9]+)\)")


def build_student_answers(
    pages: list[dict[str, Any]],
    marker_pattern: str = DEFAULT_MARKER_PATTERN,
) -> list[StudentAnswer]:
    """Build structured StudentAnswer objects from normalized pages."""
    question_map, _ = associate_regions(pages, marker_pattern)
    answers: list[StudentAnswer] = []

    for qid, regions in question_map.items():
        answers.append(build_single_student_answer(qid, regions))

    return answers


def build_single_student_answer(
    question_id: str,
    regions: list[AnswerRegion],
) -> StudentAnswer:
    """Build a StudentAnswer object from a list of AnswerRegions."""
    pages_set: set[int] = set()
    text_parts: list[str] = []
    formulas: list[Formula] = []
    tables: list[Table] = []
    charts: list[Chart] = []
    diagrams: list[Diagram] = []
    subparts_found: list[str] = []

    for idx, ans_reg in enumerate(regions):
        reg = ans_reg.region
        pages_set.add(reg.page_number)
        text = reg.text or ""

        if ans_reg.is_boundary_region:
            # Check for subpart tags
            for m in _SUBPART_RE.finditer(text):
                sub = f"({m.group('subpart')})"
                if sub not in subparts_found:
                    subparts_found.append(sub)

        # Collect text
        if text.strip():
            text_parts.append(text.strip())

        # Categorize by content_type
        c_type = reg.content_type.lower()
        if c_type == "formula":
            formulas.append(
                Formula(
                    formula_id=f"{question_id}_F{len(formulas) + 1}",
                    raw_text=text,
                    source_region_id=reg.region_id,
                    page_number=reg.page_number,
                    bbox=reg.bbox,
                )
            )
        elif c_type == "table":
            tables.append(
                Table(
                    table_id=f"{question_id}_T{len(tables) + 1}",
                    raw_text=text,
                    source_region_id=reg.region_id,
                    page_number=reg.page_number,
                    bbox=reg.bbox,
                )
            )
        elif c_type in {"chart", "figure"}:
            charts.append(
                Chart(
                    chart_id=f"{question_id}_C{len(charts) + 1}",
                    description=text,
                    source_region_id=reg.region_id,
                    page_number=reg.page_number,
                    bbox=reg.bbox,
                )
            )
        elif c_type == "diagram":
            diagrams.append(
                Diagram(
                    diagram_id=f"{question_id}_D{len(diagrams) + 1}",
                    description=text,
                    source_region_id=reg.region_id,
                    page_number=reg.page_number,
                    bbox=reg.bbox,
                )
            )

    full_text = "\n".join(text_parts)

    return StudentAnswer(
        question_id=question_id,
        pages=sorted(pages_set),
        source_regions=regions,
        answer_images=[],
        text=full_text,
        formulas=formulas,
        tables=tables,
        charts=charts,
        diagrams=diagrams,
        ocr_confidence=_estimate_ocr_confidence(full_text),
        subparts=subparts_found,
    )


def _estimate_ocr_confidence(text: str) -> float:
    """Heuristic OCR confidence estimate (0.0 to 1.0) based on character sanity."""
    if not text.strip():
        return 0.0
    # Heuristics: ratio of printable alphanumeric / standard punctuation vs gibberish
    valid_chars = sum(1 for c in text if c.isalnum() or c.isspace() or c in ".,:;()[]-+=*/'\"%_!?")
    ratio = valid_chars / max(len(text), 1)
    return round(min(1.0, max(0.1, ratio)), 2)

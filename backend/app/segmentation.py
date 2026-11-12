"""Deterministic, regex-based question segmentation for normalized OCR output."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


from backend.app.question_detector import (
    DEFAULT_MARKER_PATTERN,
    _CIRCLED_DIGIT_RE,
    _CIRCLED_TO_INT,
    _EXPLICIT_PREFIX_RE,
    _QUESTION_KEYWORDS,
    _QUESTION_KEYWORD_WINDOW,
    is_noise_line as _is_noise_line,
    is_question_header as _is_question_header,
    match_question_marker,
)


@dataclass
class SegmentationConfig:
    marker_pattern: str = DEFAULT_MARKER_PATTERN


@dataclass
class SegmentedDocument:
    document_id: str
    segmentation_path: str
    question_count: int
    cached: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def segment_normalized_document(
    normalized_path: str | Path,
    output_root: str | Path,
    config: SegmentationConfig | None = None,
) -> SegmentedDocument:
    """Segment normalized OCR with the supplied deterministic marker pattern."""
    normalized_file = Path(normalized_path)
    normalized_bytes = normalized_file.read_bytes()
    normalized_sha256 = hashlib.sha256(normalized_bytes).hexdigest()
    normalized = json.loads(normalized_bytes.decode("utf-8"))
    document_id = str(normalized["document_id"])
    output_path = Path(output_root) / _safe_document_id(document_id) / "segmentation.json"
    if _is_current_cache(output_path, document_id, normalized_sha256):
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        return SegmentedDocument(document_id, str(output_path.resolve()), len(payload["questions"]), cached=True)

    marker = re.compile((config or SegmentationConfig()).marker_pattern, re.IGNORECASE)
    questions: list[dict[str, Any]] = []
    unassigned_elements: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    seen_ids: set[str] = set()  # Track which question IDs we've seen.

    for page in normalized.get("pages", []):
        page_number = page.get("page_number")
        for element in page.get("elements", []):
            # Filter banner / noise lines before question matching.
            if element.get("type") == "text" and isinstance(element.get("text"), str):
                if _is_noise_line(element["text"]):
                    continue

            match = _question_match(element, marker, seen_ids)
            if match:
                parent_id = f"Q{match.group('number')}"
                is_subpart = bool(match.groupdict().get("subquestion"))
                # If this is a subpart of the question already being collected, fold in.
                if is_subpart and current is not None and current["question_id"] == parent_id:
                    _append_element(current, page_number, element)
                    remaining = match.group("remaining").strip()
                    if remaining:
                        current["_answer_parts"].append(remaining)
                else:
                    # New main question (or first subpart of a new parent).
                    current = _new_question(parent_id, page_number)
                    questions.append(current)
                    seen_ids.add(parent_id)
                    _append_element(current, page_number, element)
                    remaining = match.group("remaining").strip()
                    if remaining:
                        current["_answer_parts"].append(remaining)
                continue

            if current is None:
                unassigned_elements.append(_element_reference(page_number, element))
                continue
            _append_element(current, page_number, element)
            if element.get("type") == "text" and isinstance(element.get("text"), str):
                current["_answer_parts"].append(element["text"].strip())

    for question in questions:
        question["answer_text"] = "\n".join(part for part in question.pop("_answer_parts") if part)
    payload = {
        "document_id": document_id,
        "normalized_sha256": normalized_sha256,
        "questions": questions,
        "unassigned_elements": unassigned_elements,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return SegmentedDocument(document_id, str(output_path.resolve()), len(questions), cached=False)


def _question_match(
    element: dict[str, Any],
    marker: re.Pattern[str],
    seen_ids: set[str] | None = None,
) -> re.Match[str] | None:
    if element.get("type") != "text" or not isinstance(element.get("text"), str):
        return None
    return match_question_marker(element["text"], marker, seen_ids)


def _new_question(question_id: str, page_number: Any) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "answer_text": "",
        "figures": [],
        "elements": [],
        "source_pages": [page_number],
        "_answer_parts": [],
    }


def _append_element(question: dict[str, Any], page_number: Any, element: dict[str, Any]) -> None:
    reference = _element_reference(page_number, element)
    question["elements"].append(reference)
    if page_number not in question["source_pages"]:
        question["source_pages"].append(page_number)
    if _is_figure(element):
        question["figures"].append(reference)


def _element_reference(page_number: Any, element: dict[str, Any]) -> dict[str, Any]:
    return {"page_number": page_number, "element": element}


def _is_figure(element: dict[str, Any]) -> bool:
    return element.get("type") in {"figure", "table"} or element.get("label") in {
        "image",
        "figure",
        "chart",
        "table",
    }


def _is_current_cache(path: Path, document_id: str, normalized_sha256: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload["document_id"] == document_id and payload["normalized_sha256"] == normalized_sha256
    except (KeyError, TypeError, json.JSONDecodeError):
        return False


def _safe_document_id(document_id: str) -> str:
    if not document_id or Path(document_id).name != document_id:
        raise ValueError(f"Invalid document_id: {document_id}")
    return document_id

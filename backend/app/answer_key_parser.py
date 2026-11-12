"""Deterministic parsing and cached processing of answer keys."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from backend.app.dataset_adapter import Document
from backend.app.normalization import normalize_raw_ocr
from backend.app.ocr_service import LlamaCppOCRService
from backend.app.preprocessing import preprocess_document
from backend.app.segmentation import DEFAULT_MARKER_PATTERN
import re


@dataclass
class AnswerKeyParseConfig:
    """Optional marks supplied by a dataset config; absent values remain null."""

    maximum_marks: dict[str, int | float] = field(default_factory=dict)
    marker_pattern: str = DEFAULT_MARKER_PATTERN


@dataclass
class ParsedAnswerKey:
    answer_key_id: str
    answer_key_path: str
    question_count: int
    cached: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def process_answer_key_document(
    document: Document,
    output_root: str | Path,
    config: AnswerKeyParseConfig | None = None,
) -> ParsedAnswerKey:
    """Run existing PDF/image stages for one answer-key document, then parse it."""
    root = Path(output_root)
    source_types = {Path(file).suffix.lower() for file in document.files}
    if source_types - {".pdf", ".png", ".jpg", ".jpeg"}:
        raise ValueError("process_answer_key_document accepts PDF/image answer keys only")
    processed = preprocess_document(document, root / "processed")
    raw_ocr = LlamaCppOCRService().ocr_document(processed, root / "ocr")
    normalized = normalize_raw_ocr(raw_ocr.raw_ocr_path, root / "normalized")
    return parse_answer_key_file(normalized.normalized_path, root / "parsed", config, document.document_id)


def parse_answer_key_file(
    source_path: str | Path,
    output_root: str | Path,
    config: AnswerKeyParseConfig | None = None,
    answer_key_id: str | None = None,
) -> ParsedAnswerKey:
    """Parse text, structured JSON, or normalized OCR JSON into answer-key entries."""
    source = Path(source_path)
    content = source.read_bytes()
    source_sha256 = hashlib.sha256(content).hexdigest()
    settings = config or AnswerKeyParseConfig()
    config_sha256 = hashlib.sha256(_config_json(settings).encode("utf-8")).hexdigest()
    key_id = answer_key_id or source.stem
    target = Path(output_root) / _safe_id(key_id) / "answer_key.json"
    if _is_current_cache(target, key_id, source_sha256, config_sha256):
        payload = json.loads(target.read_text(encoding="utf-8"))
        return ParsedAnswerKey(key_id, str(target.resolve()), len(payload["questions"]), cached=True)

    suffix = source.suffix.lower()
    if suffix == ".txt":
        questions = _parse_text(source.read_text(encoding="utf-8"), settings)
    elif suffix == ".json":
        questions = _parse_json(content, settings)
    else:
        raise ValueError("Answer-key parsing accepts .txt, .json, or normalized OCR JSON files")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "answer_key_id": key_id,
                "source_sha256": source_sha256,
                "config_sha256": config_sha256,
                "questions": questions,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return ParsedAnswerKey(key_id, str(target.resolve()), len(questions), cached=False)


def _parse_json(content: bytes, config: AnswerKeyParseConfig) -> list[dict[str, Any]]:
    payload = json.loads(content.decode("utf-8"))
    if isinstance(payload.get("questions"), list):
        return [_structured_question(item, config) for item in payload["questions"]]
    if isinstance(payload.get("pages"), list):
        return _parse_pages(payload["pages"], config)
    raise ValueError("Structured answer-key JSON needs a questions list or normalized pages list")


def _parse_text(text: str, config: AnswerKeyParseConfig) -> list[dict[str, Any]]:
    pages = [{"page_number": 1, "elements": [{"type": "text", "text": line} for line in text.splitlines()]}]
    return _parse_pages(pages, config)


from backend.app.question_detector import (
    DEFAULT_MARKER_PATTERN,
    is_noise_line,
    match_question_marker,
)


def _parse_pages(pages: list[dict[str, Any]], config: AnswerKeyParseConfig) -> list[dict[str, Any]]:
    """Parse normalized OCR pages into answer key questions.

    Key behaviours:
    * Filters out llama-cli banner/noise lines that may survive in OCR output.
    * Handles circled-digit question markers (① ② …) produced by OCR.
    * **Merges** all fragments that share the same question_id into one entry,
      so that multi-page answers for Q1 / Q2 are not split into Q1-fragment-1,
      Q1-fragment-2, etc.  The merged order follows document order.
    """
    marker = re.compile(config.marker_pattern, re.IGNORECASE)
    # Ordered dict: question_id -> {question_id, _parts, source_pages}
    merged: dict[str, dict[str, Any]] = {}
    current_id: str | None = None

    for page in pages:
        page_number = page.get("page_number")
        for element in page.get("elements", []):
            if element.get("type") != "text":
                continue
            text = element.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            # Drop banner / noise lines.
            if is_noise_line(text):
                continue

            match = match_question_marker(text, marker)
            if match:
                parent_id = f"Q{match.group('number')}"
                is_subpart = bool(match.groupdict().get("subquestion"))

                if is_subpart and current_id == parent_id:
                    # Sub-part of the question already being collected — fold in.
                    remaining = match.group("remaining").strip()
                    if remaining:
                        merged[current_id]["_parts"].append(remaining)
                    if page_number not in merged[current_id]["source_pages"]:
                        merged[current_id]["source_pages"].append(page_number)
                else:
                    # New question (or new parent after subparts).
                    current_id = parent_id
                    if current_id not in merged:
                        merged[current_id] = {
                            "question_id": current_id,
                            "_parts": [],
                            "source_pages": [],
                        }
                    remaining = match.group("remaining").strip()
                    if remaining:
                        merged[current_id]["_parts"].append(remaining)
                    if page_number not in merged[current_id]["source_pages"]:
                        merged[current_id]["source_pages"].append(page_number)
            elif current_id is not None:
                merged[current_id]["_parts"].append(text.strip())
                if page_number not in merged[current_id]["source_pages"]:
                    merged[current_id]["source_pages"].append(page_number)

    return [
        {
            "question_id": q["question_id"],
            "answer_text": "\n".join(q["_parts"]),
            "maximum_marks": config.maximum_marks.get(q["question_id"]),
            "source_pages": q["source_pages"],
        }
        for q in merged.values()
    ]


def _structured_question(value: Any, config: AnswerKeyParseConfig) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("question_id"), str):
        raise ValueError("Each structured question requires question_id")
    answer_text = value.get("answer_text")
    if not isinstance(answer_text, str):
        raise ValueError(f"Structured question {value['question_id']} requires answer_text")
    maximum_marks = value.get("maximum_marks", config.maximum_marks.get(value["question_id"]))
    if maximum_marks is not None and (isinstance(maximum_marks, bool) or not isinstance(maximum_marks, (int, float))):
        raise ValueError(f"maximum_marks for {value['question_id']} must be numeric or null")
    source_pages = value.get("source_pages")
    item: dict[str, Any] = {
        "question_id": value["question_id"],
        "answer_text": answer_text,
        "maximum_marks": maximum_marks,
        "source_pages": source_pages if isinstance(source_pages, list) else [],
    }
    rubric = value.get("rubric") or value.get("marking_scheme")
    if isinstance(rubric, str) and rubric.strip():
        item["rubric"] = rubric.strip()
    key_concepts = value.get("key_concepts")
    if isinstance(key_concepts, list):
        item["key_concepts"] = [str(c) for c in key_concepts]
    return item


def _config_json(config: AnswerKeyParseConfig) -> str:
    return json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))


def _is_current_cache(path: Path, key_id: str, source_sha256: str, config_sha256: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return (
            payload["answer_key_id"] == key_id
            and payload["source_sha256"] == source_sha256
            and payload["config_sha256"] == config_sha256
        )
    except (KeyError, TypeError, json.JSONDecodeError):
        return False


def _safe_id(value: str) -> str:
    if not value or Path(value).name != value:
        raise ValueError(f"Invalid answer_key_id: {value}")
    return value

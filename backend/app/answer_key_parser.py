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


def _parse_pages(pages: list[dict[str, Any]], config: AnswerKeyParseConfig) -> list[dict[str, Any]]:
    marker = re.compile(config.marker_pattern, re.IGNORECASE)
    questions: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for page in pages:
        page_number = page.get("page_number")
        for element in page.get("elements", []):
            text = element.get("text")
            match = marker.match(text) if element.get("type") == "text" and isinstance(text, str) else None
            if match:
                question_id = f"Q{match.group('number')}"
                subquestion = match.groupdict().get("subquestion")
                if subquestion:
                    question_id += f"({subquestion.lower()})"
                current = {"question_id": question_id, "_parts": [], "source_pages": [page_number]}
                questions.append(current)
                remaining = match.group("remaining").strip()
                if remaining:
                    current["_parts"].append(remaining)
            elif current is not None and element.get("type") == "text" and isinstance(text, str) and text.strip():
                current["_parts"].append(text.strip())
                if page_number not in current["source_pages"]:
                    current["source_pages"].append(page_number)
    parsed = []
    for question in questions:
        question_id = question["question_id"]
        parsed.append(
            {
                "question_id": question_id,
                "answer_text": "\n".join(question["_parts"]),
                "maximum_marks": config.maximum_marks.get(question_id),
                "source_pages": question["source_pages"],
            }
        )
    return parsed


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
    return {
        "question_id": value["question_id"],
        "answer_text": answer_text,
        "maximum_marks": maximum_marks,
        "source_pages": source_pages if isinstance(source_pages, list) else [],
    }


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

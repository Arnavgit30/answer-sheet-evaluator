"""Normalize cached OCR raw output into a stable internal schema."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class NormalizedDocument:
    document_id: str
    normalized_path: str
    page_count: int
    cached: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_raw_ocr(raw_ocr_path: str | Path, output_root: str | Path) -> NormalizedDocument:
    """Read raw OCR without changing it and write a separately cached normalized file."""
    raw_path = Path(raw_ocr_path)
    raw_bytes = raw_path.read_bytes()
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    raw = json.loads(raw_bytes.decode("utf-8"))
    document_id = str(raw["document_id"])
    normalized_path = Path(output_root) / _safe_document_id(document_id) / "normalized.json"

    if _is_current_cache(normalized_path, document_id, raw_sha256):
        page_count = len(json.loads(normalized_path.read_text(encoding="utf-8"))["pages"])
        return NormalizedDocument(document_id, str(normalized_path.resolve()), page_count, cached=True)

    pages = [_normalize_page(raw_page) for raw_page in raw.get("pages", [])]
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_path.write_text(
        json.dumps(
            {
                "document_id": document_id,
                "raw_ocr_sha256": raw_sha256,
                "pages": pages,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return NormalizedDocument(document_id, str(normalized_path.resolve()), len(pages), cached=False)


def _normalize_page(raw_page: dict[str, Any]) -> dict[str, Any]:
    paddle = raw_page.get("paddle_output") or {}
    elements = [_normalize_block(block) for block in paddle.get("parsing_res_list", [])]
    return {
        "page_number": raw_page.get("page_number"),
        "source_file": raw_page.get("processed_file"),
        "width": paddle.get("width"),
        "height": paddle.get("height"),
        "elements": elements,
    }


def _normalize_block(block: dict[str, Any]) -> dict[str, Any]:
    label = block.get("block_label")
    content = block.get("block_content")
    return {
        "type": _element_type(label),
        "label": label,
        "order": block.get("block_order"),
        "text": content if isinstance(content, str) else None,
        "bbox": block.get("block_bbox"),
        "polygon": block.get("block_polygon_points"),
        "source_block_id": block.get("block_id"),
    }


def _element_type(label: object) -> str:
    value = str(label or "unknown").lower()
    if value in {"image", "figure", "chart", "table"}:
        return "figure" if value in {"image", "figure", "chart"} else "table"
    return value


def _is_current_cache(path: Path, document_id: str, raw_sha256: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload["document_id"] == document_id and payload["raw_ocr_sha256"] == raw_sha256
    except (KeyError, TypeError, json.JSONDecodeError):
        return False


def _safe_document_id(document_id: str) -> str:
    if not document_id or Path(document_id).name != document_id:
        raise ValueError(f"Invalid document_id: {document_id}")
    return document_id

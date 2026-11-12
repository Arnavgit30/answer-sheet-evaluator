"""PDF/image preprocessing with filesystem caching.

Original dataset files are read only. PDFs are rendered to PNG pages and image
files are copied into a separate processed-pages directory for OCR.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from backend.app.dataset_adapter import Document


@dataclass
class ProcessedPage:
    page_number: int
    source_file: str
    processed_file: str
    width: int | None = None
    height: int | None = None


@dataclass
class PreprocessedDocument:
    document_id: str
    pages: list[ProcessedPage]
    cached: bool

    def to_dict(self) -> dict:
        return asdict(self)


def preprocess_document(document: Document, output_root: str | Path) -> PreprocessedDocument:
    """Return OCR-ready page files, reusing them when source files are unchanged."""
    document_dir = Path(output_root) / _safe_document_id(document.document_id)
    metadata_path = document_dir / "preprocess.json"
    source_state = [_source_state(Path(file)) for file in document.files]
    cached = _load_cached(metadata_path, document.document_id, source_state)
    if cached is not None:
        return PreprocessedDocument(document.document_id, cached, cached=True)

    document_dir.mkdir(parents=True, exist_ok=True)
    pages: list[ProcessedPage] = []
    page_number = 1
    for source in map(Path, document.files):
        suffix = source.suffix.lower()
        if suffix == ".pdf":
            rendered_pages = _render_pdf(source, document_dir, page_number)
            pages.extend(rendered_pages)
            page_number += len(rendered_pages)
        elif suffix in {".png", ".jpg", ".jpeg"}:
            destination = document_dir / f"page_{page_number:03d}{suffix}"
            shutil.copy2(source, destination)
            width, height = _image_dimensions(destination)
            pages.append(ProcessedPage(page_number, str(source.resolve()), str(destination.resolve()), width, height))
            page_number += 1
        else:
            raise ValueError(f"Unsupported input file: {source}")

    payload = {
        "document_id": document.document_id,
        "source_state": source_state,
        "pages": [asdict(page) for page in pages],
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return PreprocessedDocument(document.document_id, pages, cached=False)


def _image_dimensions(image_path: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        return None, None


def _render_pdf(source: Path, destination_dir: Path, start_page: int) -> list[ProcessedPage]:
    # Import only when PDF processing is requested, keeping image-only use lightweight.
    import fitz

    pages: list[ProcessedPage] = []
    with fitz.open(source) as pdf:
        for index, pdf_page in enumerate(pdf):
            page_number = start_page + index
            destination = destination_dir / f"page_{page_number:03d}.png"
            pix = pdf_page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            pix.save(destination)
            pages.append(
                ProcessedPage(
                    page_number,
                    str(source.resolve()),
                    str(destination.resolve()),
                    width=pix.width,
                    height=pix.height,
                )
            )
    return pages


def _load_cached(metadata_path: Path, document_id: str, source_state: list[dict]) -> list[ProcessedPage] | None:
    if not metadata_path.is_file():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if payload.get("document_id") != document_id or payload.get("source_state") != source_state:
            return None
        pages = [ProcessedPage(**page) for page in payload["pages"]]
        if pages and all(Path(page.processed_file).is_file() for page in pages):
            return pages
    except (KeyError, TypeError, json.JSONDecodeError):
        pass
    return None


def _source_state(source: Path) -> dict:
    if not source.is_file():
        raise FileNotFoundError(f"Input file does not exist: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(source.resolve()), "sha256": digest.hexdigest()}


def _safe_document_id(document_id: str) -> str:
    value = str(document_id)
    if not value or Path(value).name != value:
        raise ValueError(f"Invalid document_id: {document_id}")
    return value

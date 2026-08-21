"""Cached llama.cpp OCR access for preprocessed document pages."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

from backend.app.preprocessing import PreprocessedDocument


DEFAULT_LLAMA_SERVER_URL = "http://127.0.0.1:8111"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 600
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LLAMA_CLI = os.getenv("LLAMA_CLI") or os.getenv("LLAMA_CLI_PATH") or "llama-cli"
DEFAULT_LLAMA_MODEL = _PROJECT_ROOT / "models" / "paddleocr-vl-gguf" / "PaddleOCR-VL-1.6-GGUF.gguf"
DEFAULT_LLAMA_MMPROJ = _PROJECT_ROOT / "models" / "paddleocr-vl-gguf" / "PaddleOCR-VL-1.6-GGUF-mmproj.gguf"
TRANSCRIPTION_PROMPT = (
    "Read this handwritten answer sheet image. Transcribe all visible text exactly. "
    "Preserve question numbers and describe any diagrams."
)


@dataclass
class OCRDocument:
    document_id: str
    raw_ocr_path: str
    page_count: int
    cached: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LlamaCppOCRService:
    """Send document images to the already-running local llama.cpp server."""

    def __init__(
        self,
        server_url: str | None = None,
        timeout_seconds: int | float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        cli_path: str | None = None,
        model_path: str | None = None,
        mmproj_path: str | None = None,
    ) -> None:
        self.server_url = (server_url or os.getenv("LLAMA_CPP_SERVER_URL") or DEFAULT_LLAMA_SERVER_URL).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.cli_path = cli_path or os.getenv("LLAMA_CLI") or os.getenv("LLAMA_CLI_PATH") or DEFAULT_LLAMA_CLI
        self.model_path = model_path or os.getenv("LLAMA_CPP_MODEL_PATH") or str(DEFAULT_LLAMA_MODEL)
        self.mmproj_path = mmproj_path or os.getenv("LLAMA_CPP_MMPROJ_PATH") or str(DEFAULT_LLAMA_MMPROJ)

    def ocr_document(self, document: PreprocessedDocument, output_root: str | Path) -> OCRDocument:
        """OCR all supplied pages once and cache Paddle-compatible raw output."""
        document_dir = Path(output_root) / _safe_document_id(document.document_id)
        raw_path = document_dir / "ocr_raw.json"
        state = [_page_state(page.processed_file) for page in document.pages]
        if _is_current_cache(raw_path, document.document_id, state):
            return OCRDocument(document.document_id, str(raw_path.resolve()), len(document.pages), cached=True)

        raw_pages = []
        for page, page_state in zip(document.pages, state):
            raw_pages.append(
                {
                    "page_number": page.page_number,
                    "processed_file": page.processed_file,
                    "source_sha256": page_state["sha256"],
                    "paddle_output": self._ocr_page(page.processed_file),
                }
            )

        document_dir.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(
            json.dumps(
                {"document_id": document.document_id, "source_state": state, "pages": raw_pages},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return OCRDocument(document.document_id, str(raw_path.resolve()), len(raw_pages), cached=False)

    def health_check(self) -> bool:
        """Return whether the configured llama.cpp server responds to /health."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=self.timeout_seconds)
            return 200 <= response.status_code < 300
        except requests.RequestException:
            return False

    def _ocr_page(self, page_path: str) -> dict[str, Any]:
        command = [
             self.cli_path,
            "-m", self.model_path,
            "--mmproj", self.mmproj_path,
            "--image", str(Path(page_path)),
            "-p", "OCR:",
            "-n", "512",
            "--temp", "0",
            "--single-turn",
            "--no-display-prompt",
            "--no-show-timings",
]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"llama-cli OCR request failed: {exc}") from exc
        if result.returncode != 0:
            raise RuntimeError(f"llama-cli OCR request failed with exit code {result.returncode}: {result.stderr.strip()}")
        transcription = _clean_llama_output(result.stdout)
        return _paddle_compatible_page(transcription)


# Patterns that match llama-cli startup/banner lines (never OCR content).
_LLAMA_BANNER_PREFIXES = (
    "llama_", "main:", "system_info:", "load_tensors:", "print_info:",
    "common_", "sampler", "sampling", "prompt eval", "eval time",
    "prompt processing", "build", "model", "ftype", "modalities",
    "available commands", "/exit", "/regen", "/clear", "/read",
    "/glob", "/image", "/video", "loaded media", "loading model",
    "exiting",
)
# Block-graphic chars used in the llama-cli ASCII-art logo.
_BLOCK_CHARS = re.compile(r"^[\u2580-\u259f\s]+$")
# ANSI escape codes.
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _clean_llama_output(stdout: str) -> str:
    """Remove llama-cli prompt echo, banner, diagnostics, and timing lines."""
    lines = _ANSI.sub("", stdout).splitlines()
    has_prompt = any(re.match(r"(?i)^>\s*ocr:", line.strip()) for line in lines)

    cleaned: list[str] = []
    past_prompt = False

    for raw_line in lines:
        line = raw_line.strip()
        lower = line.lower()

        if has_prompt:
            if lower.startswith("> ocr:"):
                past_prompt = True
                after = line[line.index(":") + 1:].strip()
                if after:
                    cleaned.append(after)
                continue
            if not past_prompt:
                continue
        
        if not line or lower.startswith(_LLAMA_BANNER_PREFIXES) or _BLOCK_CHARS.match(line):
            continue
        if re.match(r"^[a-z_]+\s*:\s", lower):
            continue
            
        cleaned.append(line)

    return "\n".join(cleaned)


def _paddle_compatible_page(transcription: str) -> dict[str, Any]:
    """Adapt a transcription to the raw shape consumed by normalization.py."""
    lines = transcription.splitlines() or [transcription]
    return {
        "parsing_res_list": [
            {
                "block_label": "text",
                "block_content": line,
                "block_order": index,
                "block_id": index,
                "block_bbox": None,
                "block_polygon_points": None,
            }
            for index, line in enumerate(lines)
            if line.strip()
        ]
    }


def _is_current_cache(raw_path: Path, document_id: str, state: list[dict[str, str]]) -> bool:
    if not raw_path.is_file():
        return False
    try:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        return payload["document_id"] == document_id and payload["source_state"] == state
    except (KeyError, TypeError, json.JSONDecodeError):
        return False


def _page_state(page_path: str) -> dict[str, str]:
    path = Path(page_path)
    if not path.is_file():
        raise FileNotFoundError(f"Processed page does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def _safe_document_id(document_id: str) -> str:
    value = str(document_id)
    if not value or Path(value).name != value:
        raise ValueError(f"Invalid document_id: {document_id}")
    return value

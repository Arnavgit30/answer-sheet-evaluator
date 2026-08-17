"""Run the cached local pipeline for one arbitrary student image and answer key."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from backend.app.answer_key_parser import process_answer_key_document
from backend.app.benchmarking import benchmark_grading
from backend.app.dataset_adapter import Document
from backend.app.grading import grade_from_files
from backend.app.normalization import normalize_raw_ocr
from backend.app.ocr_service import LlamaCppOCRService
from backend.app.preprocessing import preprocess_document
from backend.app.segmentation import segment_normalized_document


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one student image against one answer-key image/PDF.")
    parser.add_argument("--student-image", required=True, help="Path to a PNG, JPG, JPEG, or PDF student answer.")
    parser.add_argument("--answer-key", required=True, help="Path to a PNG, JPG, JPEG, or PDF answer key.")
    parser.add_argument("--output", default="outputs/ad_hoc", help="Cache/output directory (default: outputs/ad_hoc).")
    args = parser.parse_args()

    student_path = _input_path(args.student_image, "student image")
    answer_key_path = _input_path(args.answer_key, "answer key")
    output_root = Path(args.output)
    student = Document(_document_id("student", student_path), [str(student_path)])
    answer_key = Document(_document_id("answer_key", answer_key_path), [str(answer_key_path)])

    processed = preprocess_document(student, output_root / "processed")
    raw_ocr = LlamaCppOCRService().ocr_document(processed, output_root / "ocr")
    normalized = normalize_raw_ocr(raw_ocr.raw_ocr_path, output_root / "normalized")
    segmented = segment_normalized_document(normalized.normalized_path, output_root / "segmentation")
    parsed_key = process_answer_key_document(answer_key, output_root / "answer_keys")
    grading = grade_from_files(segmented.segmentation_path, parsed_key.answer_key_path, output_root / "grading")
    benchmark = benchmark_grading(grading.grading_path, None, output_root / "benchmark")

    result = json.loads(Path(grading.grading_path).read_text(encoding="utf-8"))
    result["benchmark"] = json.loads(Path(benchmark.benchmark_path).read_text(encoding="utf-8"))["metrics"]
    result["cache"] = {
        "student_preprocessing": processed.cached,
        "student_ocr": raw_ocr.cached,
        "student_normalization": normalized.cached,
        "student_segmentation": segmented.cached,
        "answer_key": parsed_key.cached,
        "grading": grading.cached,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _input_path(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"{label} does not exist: {path}")
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        allowed = ", ".join(sorted(IMAGE_EXTENSIONS))
        raise SystemExit(f"{label} must be one of: {allowed}")
    return path


def _document_id(prefix: str, path: Path) -> str:
    # Stable path-derived ID lets unchanged inputs reuse their own cache safely.
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


if __name__ == "__main__":
    main()

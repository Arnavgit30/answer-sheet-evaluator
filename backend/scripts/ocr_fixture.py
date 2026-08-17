"""Run one cached llama.cpp OCR check on the first configured student document."""

from __future__ import annotations

import argparse
import json

from backend.app.dataset_adapter import build_manifest
from backend.app.ocr_service import LlamaCppOCRService
from backend.app.preprocessing import preprocess_document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/dataset_config.yaml")
    parser.add_argument("--processed-output", default="outputs/processed")
    parser.add_argument("--ocr-output", default="outputs/ocr")
    args = parser.parse_args()

    manifest = build_manifest(args.config)
    document = preprocess_document(manifest.student_documents[0], args.processed_output)
    result = LlamaCppOCRService().ocr_document(document, args.ocr_output)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()

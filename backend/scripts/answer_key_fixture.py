"""Process the first configured PDF/image answer key through the existing OCR stages."""

from __future__ import annotations

import argparse
import json

from backend.app.answer_key_parser import process_answer_key_document
from backend.app.dataset_adapter import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/dataset_config.yaml")
    parser.add_argument("--output", default="outputs/answer_keys")
    args = parser.parse_args()
    manifest = build_manifest(args.config)
    result = process_answer_key_document(manifest.answer_keys[0], args.output)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()

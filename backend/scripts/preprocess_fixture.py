"""Preprocess the first discovered student document for a local Phase C check."""

from __future__ import annotations

import argparse
import json

from backend.app.dataset_adapter import build_manifest
from backend.app.preprocessing import preprocess_document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/dataset_config.yaml")
    parser.add_argument("--output", default="outputs/processed")
    args = parser.parse_args()

    manifest = build_manifest(args.config)
    result = preprocess_document(manifest.student_documents[0], args.output)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()

"""Segment the existing normalized OCR fixture without rerunning OCR."""

from __future__ import annotations

import argparse
import json

from backend.app.segmentation import segment_normalized_document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalized", default="outputs/normalized/student_001/normalized.json")
    parser.add_argument("--output", default="outputs/segmentation")
    args = parser.parse_args()
    print(json.dumps(segment_normalized_document(args.normalized, args.output).to_dict(), indent=2))


if __name__ == "__main__":
    main()

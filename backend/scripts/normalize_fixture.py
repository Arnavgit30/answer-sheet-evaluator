"""Normalize the existing single-page raw OCR fixture without rerunning OCR."""

from __future__ import annotations

import argparse
import json

from backend.app.normalization import normalize_raw_ocr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="outputs/ocr/student_001/ocr_raw.json")
    parser.add_argument("--output", default="outputs/normalized")
    args = parser.parse_args()
    print(json.dumps(normalize_raw_ocr(args.raw, args.output).to_dict(), indent=2))


if __name__ == "__main__":
    main()

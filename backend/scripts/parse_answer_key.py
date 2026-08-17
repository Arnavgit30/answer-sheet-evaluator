"""Parse a text, structured JSON, or normalized answer-key file without OCR."""

from __future__ import annotations

import argparse
import json

from backend.app.answer_key_parser import parse_answer_key_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--output", default="outputs/synthetic_answer_keys")
    parser.add_argument("--answer-key-id", default=None)
    args = parser.parse_args()
    result = parse_answer_key_file(args.source, args.output, answer_key_id=args.answer_key_id)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()

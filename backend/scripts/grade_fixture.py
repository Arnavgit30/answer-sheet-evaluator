"""Grade one pre-segmented student fixture against one parsed answer key."""

from __future__ import annotations

import argparse
import json

from backend.app.grading import grade_from_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("student_segmentation")
    parser.add_argument("answer_key")
    parser.add_argument("--output", default="outputs/grading")
    args = parser.parse_args()
    print(json.dumps(grade_from_files(args.student_segmentation, args.answer_key, args.output).to_dict(), indent=2))


if __name__ == "__main__":
    main()

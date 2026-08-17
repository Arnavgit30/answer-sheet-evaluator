"""Benchmark one grading result against optional human-marks JSON."""

from __future__ import annotations

import argparse
import json

from backend.app.benchmarking import benchmark_grading


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("grading")
    parser.add_argument("--human-marks", default=None)
    parser.add_argument("--output", default="outputs/benchmark")
    args = parser.parse_args()
    print(json.dumps(benchmark_grading(args.grading, args.human_marks, args.output).to_dict(), indent=2))


if __name__ == "__main__":
    main()

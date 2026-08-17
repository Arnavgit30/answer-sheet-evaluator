"""Optional benchmark metrics comparing predicted marks with human marks."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkResult:
    document_id: str
    benchmark_path: str
    cached: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def benchmark_grading(
    grading_path: str | Path, human_marks_path: str | Path | None, output_root: str | Path
) -> BenchmarkResult:
    """Calculate metrics only for questions with valid predicted and human marks."""
    grading_file = Path(grading_path)
    grading_bytes = grading_file.read_bytes()
    grading_sha256 = hashlib.sha256(grading_bytes).hexdigest()
    grading = json.loads(grading_bytes.decode("utf-8"))
    document_id = str(grading["document_id"])
    human_marks, human_sha256 = _load_human_marks(human_marks_path)
    target = Path(output_root) / _safe_document_id(document_id) / "benchmark.json"
    if _is_current_cache(target, document_id, grading_sha256, human_sha256):
        return BenchmarkResult(document_id, str(target.resolve()), cached=True)

    pairs = []
    for result in grading.get("questions", []):
        question_id = result.get("question_id")
        predicted = result.get("predicted_marks")
        human = human_marks.get(question_id)
        if _number(predicted) and _number(human):
            pairs.append({"question_id": question_id, "predicted_marks": float(predicted), "human_marks": float(human)})
    metrics = compute_metrics(pairs)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "document_id": document_id,
                "grading_sha256": grading_sha256,
                "human_marks_sha256": human_sha256,
                "metrics": metrics,
                "evaluated_questions": pairs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return BenchmarkResult(document_id, str(target.resolve()), cached=False)


def compute_metrics(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Return only mathematically valid metrics; unavailable values are null."""
    if not pairs:
        return {
            "status": "ground_truth_unavailable",
            "evaluated_question_count": 0,
            "mae": None,
            "rmse": None,
            "pearson_correlation": None,
            "spearman_correlation": None,
            "agreement_within_1_mark": None,
        }
    predicted = [float(pair["predicted_marks"]) for pair in pairs]
    human = [float(pair["human_marks"]) for pair in pairs]
    errors = [abs(value - truth) for value, truth in zip(predicted, human)]
    pearson = _correlation(predicted, human) if len(pairs) >= 2 else None
    spearman = _correlation(_ranks(predicted), _ranks(human)) if len(pairs) >= 2 else None
    return {
        "status": "ground_truth_available",
        "evaluated_question_count": len(pairs),
        "mae": sum(errors) / len(errors),
        "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
        "pearson_correlation": pearson,
        "spearman_correlation": spearman,
        "agreement_within_1_mark": sum(error <= 1.0 for error in errors) / len(errors),
    }


def _load_human_marks(path: str | Path | None) -> tuple[dict[str, float], str | None]:
    if path is None:
        return {}, None
    source = Path(path)
    content = source.read_bytes()
    payload = json.loads(content.decode("utf-8"))
    entries = payload.get("marks", payload) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError("Human marks JSON must be a marks list")
    marks: dict[str, float] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("question_id"), str):
            raise ValueError("Each human-mark entry requires question_id")
        value = entry.get("human_marks")
        if not _number(value):
            raise ValueError(f"human_marks for {entry['question_id']} must be numeric")
        marks[entry["question_id"]] = float(value)
    return marks, hashlib.sha256(content).hexdigest()


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _correlation(left: list[float], right: list[float]) -> float | None:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_scale == 0 or right_scale == 0:
        return None
    return numerator / (left_scale * right_scale)


def _ranks(values: list[float]) -> list[float]:
    ranks = [0.0] * len(values)
    ordered = sorted(enumerate(values), key=lambda pair: pair[1])
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for original_index, _ in ordered[index:end]:
            ranks[original_index] = rank
        index = end
    return ranks


def _is_current_cache(path: Path, document_id: str, grading_sha256: str, human_sha256: str | None) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return (
            payload["document_id"] == document_id
            and payload["grading_sha256"] == grading_sha256
            and payload["human_marks_sha256"] == human_sha256
        )
    except (KeyError, TypeError, json.JSONDecodeError):
        return False


def _safe_document_id(document_id: str) -> str:
    if not document_id or Path(document_id).name != document_id:
        raise ValueError(f"Invalid document_id: {document_id}")
    return document_id

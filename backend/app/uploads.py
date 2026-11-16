"""Save uploaded files as a one-student dataset the existing pipeline can read."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_FILE_BYTES = 25 * 1024 * 1024
STUDENT_EXTS = {".pdf", ".png", ".jpg", ".jpeg"}
KEY_EXTS = STUDENT_EXTS | {".txt", ".json"}


def save_upload(student: list[tuple[str, bytes]], key: list[tuple[str, bytes]]) -> dict[str, str]:
    """student/key: (original_filename, bytes). Returns project-relative config_path + student id."""
    _check(student, STUDENT_EXTS, "student")
    _check(key, KEY_EXTS, "answer key")
    if len(key) > 1 and any(_ext(n) in {".txt", ".json"} for n, _ in key):
        raise ValueError("A .txt/.json answer key must be a single file")

    student_id, key_id = f"stu_{_digest(student)}", f"key_{_digest(key)}"
    root = PROJECT_ROOT / "data" / "uploads" / f"{student_id}_{key_id}"
    _write(root / "student_answers", student_id, student)
    _write(root / "answer_keys", key_id, key)
    config = root / "configs" / "dataset_config.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        json.dumps(
            {
                "dataset_name": root.name,
                "student_answers": {"path": "student_answers"},
                "answer_keys": {"path": "answer_keys"},
            }
        ),
        encoding="utf-8",
    )
    return {
        "config_path": config.relative_to(PROJECT_ROOT).as_posix(),
        "student_document_id": student_id,
    }


def _ext(name: str) -> str:
    return Path(name or "").suffix.lower()


def _check(files: list[tuple[str, bytes]], allowed: set[str], label: str) -> None:
    if not files:
        raise ValueError(f"No {label} file uploaded")
    for name, data in files:
        if _ext(name) not in allowed:
            raise ValueError(f"Unsupported {label} file type: {_ext(name) or 'none'}")
        if not data:
            raise ValueError(f"Empty {label} file")
        if len(data) > MAX_FILE_BYTES:
            raise ValueError(f"{label} file exceeds {MAX_FILE_BYTES // (1024 * 1024)} MB")


def _digest(files: list[tuple[str, bytes]]) -> str:
    h = hashlib.sha256()
    for _, data in files:
        h.update(hashlib.sha256(data).digest())
    return h.hexdigest()[:12]


def _write(folder: Path, doc_id: str, files: list[tuple[str, bytes]]) -> None:
    if len(files) == 1:
        target = folder / f"{doc_id}{_ext(files[0][0])}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(files[0][1])
        return
    for i, (name, data) in enumerate(files, start=1):
        target = folder / doc_id / f"page_{i:03d}{_ext(name)}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

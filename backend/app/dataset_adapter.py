"""Configuration-driven discovery of answer-sheet datasets.

Only this module knows about dataset directory layouts. Later pipeline stages use
the manifest produced here and do not access configured dataset paths directly.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SUPPORTED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}
SUPPORTED_ANSWER_KEY_EXTENSIONS = SUPPORTED_EXTENSIONS | {"txt", "json"}


@dataclass
class Document:
    document_id: str
    files: list[str]
    answer_key_id: str | None = None
    human_marks_available: bool = False


@dataclass
class DatasetManifest:
    dataset_id: str
    student_documents: list[Document]
    answer_keys: list[Document]
    human_marks_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatasetValidationError(ValueError):
    """Raised when a configuration cannot produce a usable dataset manifest."""


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Read JSON or the small YAML subset used by the demo configuration."""
    path = Path(config_path)
    if not path.is_file():
        raise DatasetValidationError(f"Dataset config does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _parse_simple_yaml(text)


def build_manifest(config_path: str | Path) -> DatasetManifest:
    config_file = Path(config_path).resolve()
    config = load_config(config_file)
    dataset_id = str(config.get("dataset_name") or config.get("dataset_id") or "dataset")
    base_dir = config_file.parent.parent

    student_config = _required_mapping(config, "student_answers")
    key_config = _required_mapping(config, "answer_keys")
    student_extensions = _extensions(student_config.get("extensions"), SUPPORTED_EXTENSIONS)
    key_extensions = _extensions(key_config.get("extensions"), SUPPORTED_ANSWER_KEY_EXTENSIONS)
    students = _discover_documents(_resolve_path(base_dir, student_config.get("path")), student_extensions)
    answer_keys = _discover_documents(_resolve_path(base_dir, key_config.get("path")), key_extensions)

    if not students:
        raise DatasetValidationError("No supported student answer files were found")
    if not answer_keys:
        raise DatasetValidationError("No supported answer key files were found")

    configured_key = student_config.get("answer_key_id") or config.get("default_answer_key_id")
    key_ids = {key.document_id for key in answer_keys}
    if configured_key and configured_key not in key_ids:
        raise DatasetValidationError(f"Configured answer_key_id was not found: {configured_key}")
    if not configured_key and len(answer_keys) == 1:
        configured_key = answer_keys[0].document_id
    if not configured_key:
        raise DatasetValidationError(
            "Multiple answer keys found; set default_answer_key_id in the dataset config"
        )

    human_config = config.get("human_marks") or {}
    if not isinstance(human_config, dict):
        raise DatasetValidationError("human_marks must be a mapping")
    marks_available = bool(human_config.get("available", False))
    marks_path = human_config.get("path")
    if marks_available:
        if not marks_path:
            raise DatasetValidationError("human_marks.path is required when human_marks.available is true")
        marks_file = _resolve_path(base_dir, marks_path)
        if not marks_file.is_file():
            raise DatasetValidationError(f"Human marks file does not exist: {marks_file}")
        resolved_marks_path: str | None = str(marks_file)
    else:
        resolved_marks_path = None

    for student in students:
        student.answer_key_id = str(configured_key)
        student.human_marks_available = marks_available
    return DatasetManifest(dataset_id, students, answer_keys, resolved_marks_path)


def validate_dataset(config_path: str | Path) -> dict[str, Any]:
    """Return a small API-friendly validation result without writing any files."""
    try:
        manifest = build_manifest(config_path)
    except DatasetValidationError as exc:
        return {"valid": False, "errors": [str(exc)], "manifest": None}
    return {"valid": True, "errors": [], "manifest": manifest.to_dict()}


def _required_mapping(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict) or not value.get("path"):
        raise DatasetValidationError(f"{name}.path is required")
    return value


def _resolve_path(base_dir: Path, value: object) -> Path:
    path = Path(str(value))
    path = path if path.is_absolute() else base_dir / path
    if not path.exists():
        raise DatasetValidationError(f"Configured path does not exist: {path}")
    return path.resolve()


def _extensions(values: object, supported: set[str]) -> set[str]:
    if values is None:
        return supported
    if not isinstance(values, list):
        raise DatasetValidationError("extensions must be a list")
    extensions = {str(value).lower().lstrip(".") for value in values}
    unsupported = extensions - supported
    if unsupported:
        raise DatasetValidationError(f"Unsupported file extensions: {', '.join(sorted(unsupported))}")
    return extensions


def _discover_documents(root: Path, extensions: set[str]) -> list[Document]:
    if root.is_file():
        return [_document(root.stem, [root], extensions)]
    documents: list[Document] = []
    for child in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if child.is_file() and _is_supported(child, extensions):
            documents.append(_document(child.stem, [child], extensions))
        elif child.is_dir():
            pages = [item for item in sorted(child.rglob("*")) if item.is_file() and _is_supported(item, extensions)]
            if pages:
                documents.append(_document(child.name, pages, extensions))
    return documents


def _document(document_id: str, files: list[Path], extensions: set[str]) -> Document:
    if not files or any(not _is_supported(file, extensions) for file in files):
        raise DatasetValidationError(f"Document {document_id} has no supported files")
    return Document(document_id=document_id, files=[str(file.resolve()) for file in files])


def _is_supported(path: Path, extensions: set[str]) -> bool:
    return path.suffix.lower().lstrip(".") in extensions


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse mappings, scalar values, and inline lists without an extra package."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        match = re.match(r"^\s*([^:]+):(?:\s*(.*))?$", line)
        if not match:
            raise DatasetValidationError(f"Unsupported YAML syntax on line {number}")
        key, raw_value = match.groups()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        key = key.strip()
        if raw_value is None or raw_value == "":
            value: Any = {}
            parent[key] = value
            stack.append((indent, value))
        else:
            parent[key] = _yaml_scalar(raw_value.strip())
    return root


def _yaml_scalar(value: str) -> Any:
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        return [item.strip().strip("'\"") for item in value[1:-1].split(",") if item.strip()]
    return value.strip("'\"")

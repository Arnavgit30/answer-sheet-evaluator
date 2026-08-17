"""Filesystem persistence for demo evaluation records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class EvaluationStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def save(self, evaluation: dict[str, Any]) -> None:
        evaluation_id = evaluation["evaluation_id"]
        path = self.root / f"{evaluation_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, evaluation_id: str) -> dict[str, Any] | None:
        path = self.root / f"{evaluation_id}.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

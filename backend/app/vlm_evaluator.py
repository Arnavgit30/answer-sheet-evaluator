"""VLM Evaluator (Qwen3.8-27B on Groq / OpenAI-compatible API) for intelligent grading."""

from __future__ import annotations

import base64
import json
import math
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

from backend.app.models import EvaluationResult, ReferenceAnswer, StudentAnswer


VLM_SYSTEM_PROMPT = (
    "You are an expert exam grader evaluating student handwritten answers against a reference key. "
    "Assess conceptual correctness, logical/mathematical reasoning, relevant intermediate steps, "
    "missing concepts, and valid alternative approaches or solutions. "
    "Award full or partial credit based on the actual understanding and marking scheme—never grade "
    "by literal word-for-word similarity. Valid alternative methods must receive full credit. "
    "Always output valid JSON with keys: "
    "marks_awarded (float between 0 and maximum_marks), "
    "reasoning (string), "
    "concepts_present (list of strings), "
    "concepts_missing (list of strings), "
    "errors (list of strings), "
    "grading_confidence (float between 0.0 and 1.0), "
    "human_review_required (boolean)."
)

MAX_IMAGES_PER_REQUEST = 3


def _load_dotenv() -> dict[str, str]:
    """Load key-value pairs from project .env files without overriding set env vars."""
    root = Path(__file__).resolve().parents[2]
    loaded: dict[str, str] = {}
    for env_path in (root / ".env", root / "backend" / ".env"):
        if not env_path.is_file():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip("'").strip('"')
            if key and val:
                loaded[key] = val
                if not os.environ.get(key):
                    os.environ[key] = val
    return loaded


class VLMEvaluator:
    """Evaluates student answers against reference keys using a Vision-Language Model."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        env_values = _load_dotenv()
        self.model = model or os.getenv("VLM_MODEL") or env_values.get("VLM_MODEL") or "qwen/qwen3.8-27b"
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = (
                os.getenv("VLM_API_KEY")
                or os.getenv("GROQ_API_KEY")
                or env_values.get("VLM_API_KEY")
                or env_values.get("GROQ_API_KEY")
                or ""
            )
        raw_url = (
            base_url
            or os.getenv("VLM_BASE_URL")
            or env_values.get("VLM_BASE_URL")
            or "https://api.groq.com/openai/v1"
        )
        self.base_url = raw_url.rstrip("/")

    def evaluate_question(
        self,
        student: StudentAnswer,
        reference: ReferenceAnswer,
        baseline_similarity: float | None = None,
        baseline_marks: float | None = None,
    ) -> EvaluationResult:
        """Grade a single question using the primary LLM/VLM without similarity fallback."""
        max_marks = float(reference.maximum_marks if reference.maximum_marks is not None else 5.0)

        # Blank student answer check: award 0 marks and flag for human review
        if _is_blank_student_answer(student):
            return EvaluationResult(
                question_id=student.question_id,
                marks_awarded=0.0,
                max_marks=max_marks,
                reasoning="No student answer detected in segmented regions — human review required!",
                concepts_present=[],
                concepts_missing=list(reference.key_concepts) if reference.key_concepts else [],
                errors=["Blank or unsegmented student answer"],
                grading_confidence=0.0,
                grading_method=self.model,
                baseline_similarity=baseline_similarity,
                baseline_marks=None,
                human_review_required=True,
                status="completed",
            )

        # Missing API key: mark pending and flag for human review (never fall back to similarity)
        if not self.api_key:
            return EvaluationResult(
                question_id=student.question_id,
                marks_awarded=None,
                max_marks=max_marks,
                reasoning="LLM evaluation pending: VLM_API_KEY is not configured. Human review required.",
                concepts_present=[],
                concepts_missing=[],
                errors=["Missing VLM_API_KEY"],
                grading_confidence=0.0,
                grading_method="llm_pending",
                baseline_similarity=baseline_similarity,
                baseline_marks=None,
                human_review_required=True,
                status="pending",
            )

        prompt = self._build_prompt(student, reference, max_marks)

        try:
            import requests

            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": VLM_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            parsed = self._clean_and_parse_json(content)
            return self._validate_and_build_result(
                parsed,
                question_id=student.question_id,
                max_marks=max_marks,
                baseline_similarity=baseline_similarity,
            )
        except Exception as exc:
            return EvaluationResult(
                question_id=student.question_id,
                marks_awarded=None,
                max_marks=max_marks,
                reasoning=f"LLM evaluation failed ({exc}). Human review required.",
                concepts_present=[],
                concepts_missing=[],
                errors=[str(exc)],
                grading_confidence=0.0,
                grading_method="llm_failed",
                baseline_similarity=baseline_similarity,
                baseline_marks=None,
                human_review_required=True,
                status="failed",
            )

    def _build_prompt(
        self,
        student: StudentAnswer,
        reference: ReferenceAnswer,
        max_marks: float,
    ) -> str:
        sections = [
            f"Question ID: {student.question_id}",
            f"Maximum Marks: {max_marks}",
            "",
            "--- REFERENCE ANSWER ---",
            reference.answer_text or "(No reference answer text provided)",
        ]
        if reference.rubric:
            sections.extend(["", "--- MARKING SCHEME / RUBRIC ---", reference.rubric])
        if reference.key_concepts:
            sections.extend(
                ["", "--- KEY CONCEPTS ---"] + [f"- {concept}" for concept in reference.key_concepts]
            )
        sections.extend(["", "--- STUDENT ANSWER ---", student.text or "(No transcribed text)"])
        if student.formulas:
            sections.append("\nExtracted Formulas:")
            sections.extend(f"- {f.raw_text}" for f in student.formulas if f.raw_text)
        if student.tables:
            sections.append("\nExtracted Tables:")
            sections.extend(f"- {t.raw_text}" for t in student.tables if t.raw_text)
        if student.charts:
            sections.append("\nExtracted Charts / Figures:")
            sections.extend(f"- {c.description}" for c in student.charts if c.description)
        if student.diagrams:
            sections.append("\nExtracted Diagrams:")
            sections.extend(f"- {d.description}" for d in student.diagrams if d.description)
        sections.extend(
            [
                "",
                "Evaluate conceptual accuracy, reasoning steps, and valid alternative approaches. "
                f"Return JSON with marks_awarded between 0 and {max_marks}.",
            ]
        )
        return "\n".join(sections)

    def _build_user_content(self, prompt: str, answer_images: list[str]) -> str | list[dict[str, Any]]:
        image_blocks: list[dict[str, Any]] = []
        for img_path_str in (answer_images or [])[:MAX_IMAGES_PER_REQUEST]:
            img_path = Path(img_path_str)
            if not img_path.is_file():
                continue
            mime_type, _ = mimetypes.guess_type(img_path.name)
            mime_type = mime_type or "image/png"
            encoded = base64.b64encode(img_path.read_bytes()).decode("utf-8")
            image_blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                }
            )
        if not image_blocks:
            return prompt
        return [{"type": "text", "text": prompt}, *image_blocks]

    def _validate_and_build_result(
        self,
        parsed: dict[str, Any],
        question_id: str,
        max_marks: float,
        baseline_similarity: float | None,
    ) -> EvaluationResult:
        if not isinstance(parsed, dict):
            raise ValueError("LLM response is not a JSON object")

        raw_marks = parsed.get("marks_awarded")
        if raw_marks is None or isinstance(raw_marks, bool) or not isinstance(raw_marks, (int, float)):
            raise ValueError(f"Invalid non-numeric marks_awarded: {raw_marks!r}")
        marks = float(raw_marks)
        if not math.isfinite(marks) or marks < 0.0 or marks > max_marks:
            raise ValueError(f"marks_awarded {marks} out of valid range [0.0, {max_marks}]")
        marks = round(marks, 2)

        raw_conf = parsed.get("grading_confidence", 0.85)
        if isinstance(raw_conf, bool) or not isinstance(raw_conf, (int, float)):
            raise ValueError(f"Invalid non-numeric grading_confidence: {raw_conf!r}")
        confidence = float(raw_conf)
        if not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
            raise ValueError(f"grading_confidence {confidence} out of valid range [0.0, 1.0]")
        confidence = round(confidence, 2)

        reasoning = str(parsed.get("reasoning") or "").strip()
        if not reasoning:
            raise ValueError("LLM response missing non-empty reasoning")

        concepts_present = _string_list(parsed.get("concepts_present", []), "concepts_present")
        concepts_missing = _string_list(parsed.get("concepts_missing", []), "concepts_missing")
        errors = _string_list(parsed.get("errors", []), "errors")

        raw_review_req = parsed.get("human_review_required", False)
        if not isinstance(raw_review_req, bool):
            raise ValueError(f"Invalid non-boolean human_review_required: {raw_review_req!r}")
        human_review_required = raw_review_req or confidence < 0.7

        return EvaluationResult(
            question_id=question_id,
            marks_awarded=marks,
            max_marks=max_marks,
            reasoning=reasoning,
            concepts_present=concepts_present,
            concepts_missing=concepts_missing,
            errors=errors,
            grading_confidence=confidence,
            grading_method=self.model,
            baseline_similarity=baseline_similarity,
            baseline_marks=None,
            human_review_required=human_review_required,
            status="completed",
        )

    def _clean_and_parse_json(self, raw: str) -> dict[str, Any]:
        cleaned = (raw or "").strip()
        # Strip optional <think>...</think> reasoning blocks if produced by Qwen models
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
        if "```" in cleaned:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1)
        return json.loads(cleaned)


def _is_blank_student_answer(student: StudentAnswer) -> bool:
    if (student.text or "").strip():
        return False
    if any((f.raw_text or "").strip() for f in student.formulas):
        return False
    if any((t.raw_text or "").strip() for t in student.tables):
        return False
    if any((c.description or "").strip() for c in student.charts):
        return False
    if any((d.description or "").strip() for d in student.diagrams):
        return False
    return True


def _string_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of strings")
    return [str(item) for item in value]

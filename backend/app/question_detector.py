"""Shared question boundary detection for segmentation and answer key parsing."""

from __future__ import annotations

import re
from typing import Any

from backend.app.models import QuestionBoundary

DEFAULT_MARKER_PATTERN = (
    r"^\s*(?:question|ques|qn|q|on|ans(?:wer)?|a)?"
    r"\s*[.:]?\s*"
    r"(?P<number>\d+)"
    r"\s*(?:\(\s*(?P<subquestion>[A-Za-z])\s*\))?"
    r"\s*(?:[.:\-\)])?"
    r"(?=\s|$)(?P<remaining>.*)$"
)

# Unicode circled digits ① (U+2460) through ⑳ (U+2473),
# parenthesized ⑴ (U+2474) through ⒇ (U+2487),
# and dotted ⒈ (U+2488) through ⒑ (U+2491).
_CIRCLED_TO_INT: dict[str, int] = {
    **{chr(0x2460 + i): i + 1 for i in range(20)},
    **{chr(0x2474 + i): i + 1 for i in range(20)},
    **{chr(0x2488 + i): i + 1 for i in range(10)},
}
_CIRCLED_RE = re.compile(
    r"^\s*(?P<circled>[\u2460-\u2473\u2474-\u247d\u2488-\u2491])\s*(?P<remaining>.*)$"
)
_CIRCLED_DIGIT_RE = _CIRCLED_RE

# Strong instruction/question keywords unlikely to appear at the START of a body paragraph.
_QUESTION_KEYWORDS = re.compile(
    r"\b(how|what|why|explain|describe|discuss|analyse|analyze"
    r"|compare|define|evaluate|elaborate|differentiate|illustrate|justify|outline"
    r"|mention|write|prove|derive|calculate|solve)\b",
    re.IGNORECASE,
)
_QUESTION_KEYWORD_WINDOW = 60

# Prefixes that unambiguously indicate a question/answer marker.
_EXPLICIT_PREFIX_RE = re.compile(
    r"^\s*(?:question|ques|qn|q|on|ans(?:wer)?|a)\s*[.:]?\s*\d",
    re.IGNORECASE,
)

# Banner / noise lines from llama that may survive in normalized OCR output.
_NOISE_PREFIXES = (
    "loading model", "exiting", "build", "ftype", "modalities",
    "available commands", "/exit", "/regen", "/clear", "/read",
    "/glob", "/image", "/video", "loaded media",
)
_BLOCK_CHARS_RE = re.compile(r"^[\u2580-\u259f\s]+$")
_KV_DIAG_RE = re.compile(r"^[a-z_]+\s*:\s")


def is_noise_line(text: str) -> bool:
    """Return True if a text line is a banner/diagnostic artifact."""
    stripped = text.strip()
    if not stripped:
        return True
    lower = stripped.lower()
    if lower.startswith(_NOISE_PREFIXES):
        return True
    if _BLOCK_CHARS_RE.match(stripped):
        return True
    if _KV_DIAG_RE.match(lower):
        return True
    if lower.startswith("> ocr:") or lower == "ocr:":
        return True
    return False


def is_question_header(
    match: re.Match[str],
    original_text: str,
    seen_ids: set[str] | None = None,
) -> bool:
    """Return True if this marker looks like a real exam question boundary."""
    remaining = match.group("remaining").strip()
    # Rule 1: explicit prefix
    if _EXPLICIT_PREFIX_RE.match(original_text):
        return True
    # Rule 2: circled digit
    if _CIRCLED_RE.match(original_text):
        return True
    # Rule 3: sub-question letter
    if match.groupdict().get("subquestion"):
        return True
    # Rule 4: first-ever occurrence of this question number for bare numbers
    # without keywords, only if marked with parenthesis (e.g. 1) or 2))
    # and strictly following sequence.
    if seen_ids is not None:
        candidate_num = int(match.group("number"))
        candidate_id = f"Q{candidate_num}"
        if candidate_id not in seen_ids:
            has_marker = bool(re.match(r"^\s*\(?\d+[.:\)]", original_text))
            if not seen_ids:
                if candidate_num <= 20 and (has_marker or candidate_num == 1):
                    return True
            else:
                max_seen = max(int(qid[1:]) for qid in seen_ids if qid[1:].isdigit())
                if candidate_num == max_seen + 1 and has_marker:
                    return True
    # Rule 5: question keyword in first window chars
    if _QUESTION_KEYWORDS.search(remaining[:_QUESTION_KEYWORD_WINDOW]):
        return True
    return False


def match_question_marker(
    text: str,
    marker: re.Pattern[str],
    seen_ids: set[str] | None = None,
) -> re.Match[str] | None:
    """Match a text line against question marker patterns, handling circled digits."""
    circled = _CIRCLED_RE.match(text)
    if circled:
        num = _CIRCLED_TO_INT.get(circled.group("circled"))
        if num is not None:
            m = marker.match(f"{num} {circled.group('remaining')}")
            if m and is_question_header(m, text, seen_ids):
                return m
    m = marker.match(text)
    if m and is_question_header(m, text, seen_ids):
        return m
    return None

"""Map each OCR element across document pages to exactly one main question."""

from __future__ import annotations

import re
from typing import Any

from backend.app.models import AnswerRegion, OCRRegion
from backend.app.question_detector import (
    DEFAULT_MARKER_PATTERN,
    is_noise_line,
    match_question_marker,
)


def associate_regions(
    pages: list[dict[str, Any]],
    marker_pattern: str = DEFAULT_MARKER_PATTERN,
) -> tuple[dict[str, list[AnswerRegion]], list[AnswerRegion]]:
    """Partition all OCR elements into per-question AnswerRegions.

    Returns:
        (question_regions_map, unassigned_regions)
    """
    marker = re.compile(marker_pattern, re.IGNORECASE)
    question_map: dict[str, list[AnswerRegion]] = {}
    unassigned: list[AnswerRegion] = []
    current_qid: str | None = None
    seen_ids: set[str] = set()

    for page in pages:
        page_num = page.get("page_number", 1)
        elements = page.get("elements", [])

        for idx, elem in enumerate(elements):
            text = elem.get("text")
            if isinstance(text, str) and is_noise_line(text):
                continue

            region_id = f"P{page_num}_E{idx + 1}"
            ocr_region = OCRRegion(
                region_id=region_id,
                page_number=page_num,
                content_type=elem.get("type", "text"),
                text=text if isinstance(text, str) else None,
                raw_content=text if isinstance(text, str) else None,
                bbox=elem.get("bbox"),
                polygon=elem.get("polygon"),
                source_block_id=elem.get("source_block_id"),
                order=elem.get("order", idx + 1),
                label=elem.get("label"),
            )

            match = match_question_marker(text, marker, seen_ids) if isinstance(text, str) else None
            if match:
                parent_id = f"Q{match.group('number')}"
                is_subpart = bool(match.groupdict().get("subquestion"))

                if is_subpart and current_qid == parent_id:
                    # Subpart within current question
                    ans_region = AnswerRegion(
                        question_id=current_qid,
                        region=ocr_region,
                        is_boundary_region=True,
                    )
                    question_map[current_qid].append(ans_region)
                else:
                    # New question boundary
                    current_qid = parent_id
                    seen_ids.add(current_qid)
                    if current_qid not in question_map:
                        question_map[current_qid] = []
                    ans_region = AnswerRegion(
                        question_id=current_qid,
                        region=ocr_region,
                        is_boundary_region=True,
                    )
                    question_map[current_qid].append(ans_region)
                continue

            if current_qid is None:
                unassigned.append(AnswerRegion(question_id="", region=ocr_region))
            else:
                ans_region = AnswerRegion(
                    question_id=current_qid,
                    region=ocr_region,
                    is_boundary_region=False,
                )
                question_map[current_qid].append(ans_region)

    return question_map, unassigned

"""Regression tests for student answer segmentation.

Covers:
- Banner/noise line filtering
- Question header heuristic (first-occurrence vs sub-section disambiguation)
- Expanded marker prefixes (Ans, A, Qn, Ques, etc.)
- C3-like OCR misread prevention (bare '17' inside Q2 body → not a new Q17)
- C4-like answer-title markers ('2) POBHC...' without question keywords)
"""

import json
import re
import pytest

from backend.app.segmentation import (
    DEFAULT_MARKER_PATTERN,
    _is_noise_line,
    _is_question_header,
    _question_match,
    segment_normalized_document,
)


@pytest.fixture
def marker():
    return re.compile(DEFAULT_MARKER_PATTERN, re.IGNORECASE)


# ── Banner / noise filtering ────────────────────────────────────────

class TestNoiseFiltering:
    def test_loading_model(self):
        assert _is_noise_line("Loading model...")

    def test_exiting(self):
        assert _is_noise_line("Exiting...")

    def test_build_line(self):
        assert _is_noise_line("build      : b10424-2bacf9ea5")

    def test_ftype_line(self):
        assert _is_noise_line("ftype      : BF16")

    def test_block_chars(self):
        assert _is_noise_line("\u2584\u2584 \u2584\u2584")
        assert _is_noise_line("\u2588\u2588 \u2588\u2588")

    def test_ocr_prompt(self):
        assert _is_noise_line("> OCR:")
        assert _is_noise_line("ocr:")

    def test_real_content(self):
        assert not _is_noise_line("VIT University can implement cluster computing.")
        assert not _is_noise_line("1) How can VIT university implement...")
        assert not _is_noise_line("Private cloud:")


# ── Marker pattern matching ─────────────────────────────────────────

class TestMarkerPattern:
    """Test that DEFAULT_MARKER_PATTERN matches various question/answer prefixes."""

    def test_bare_number(self, marker):
        m = marker.match("1) How can VIT")
        assert m and m.group("number") == "1"

    def test_q_prefix(self, marker):
        m = marker.match("Q1. Implementation")
        assert m and m.group("number") == "1"

    def test_question_prefix(self, marker):
        m = marker.match("Question 2: Explain cloud")
        assert m and m.group("number") == "2"

    def test_qn_prefix(self, marker):
        m = marker.match("Qn.2. Cloud computing")
        assert m and m.group("number") == "2"

    def test_on_prefix(self, marker):
        """Test OCR misreading 'Qn' as 'On'."""
        m = marker.match("On.1 Implementation")
        assert m and m.group("number") == "1"

    def test_ques_prefix(self, marker):
        m = marker.match("Ques 1 Implementation")
        assert m and m.group("number") == "1"

    def test_ans_prefix(self, marker):
        m = marker.match("Ans 1. Implementation of cluster")
        assert m and m.group("number") == "1"

    def test_answer_prefix(self, marker):
        m = marker.match("Answer 2 Cloud computing")
        assert m and m.group("number") == "2"

    def test_a_prefix(self, marker):
        m = marker.match("A1 Cloud computing")
        assert m and m.group("number") == "1"

    def test_subquestion(self, marker):
        m = marker.match("1(a) First part")
        assert m and m.group("number") == "1" and m.group("subquestion") == "a"


# ── Question header heuristic ───────────────────────────────────────

class TestQuestionHeaderHeuristic:
    """Test _is_question_header correctly distinguishes real question markers
    from numbered body sub-sections."""

    def test_explicit_q_prefix(self, marker):
        m = marker.match("1 Implementation of cluster")
        assert m
        assert _is_question_header(m, "Q1. Implementation of cluster")

    def test_explicit_qn_prefix(self, marker):
        m = marker.match("2 Cloud computing")
        assert m
        assert _is_question_header(m, "Qn.2. Cloud computing")

    def test_explicit_ans_prefix(self, marker):
        m = marker.match("1 Implementation")
        assert m
        assert _is_question_header(m, "Ans 1. Implementation")

    def test_question_keyword(self, marker):
        m = marker.match("1 How can VIT university implement")
        assert m
        assert _is_question_header(m, "1 How can VIT university implement")

    def test_first_occurrence_accepted(self, marker):
        """First-ever occurrence of a new number should be accepted even without
        keywords -- this is the C4 fix ('2) POBHC...' has no question keyword)."""
        m = marker.match("2 POBHC, Private, Hybrid Cloud")
        assert m
        assert _is_question_header(m, "2) POBHC, Private, Hybrid Cloud", seen_ids=set())

    def test_second_occurrence_rejected_without_keyword(self, marker):
        """Second occurrence of same number WITHOUT keyword should be rejected."""
        m = marker.match("2 Parallel Processing")
        assert m
        assert not _is_question_header(m, "2 Parallel Processing", seen_ids={"Q1", "Q2"})

    def test_c3_q17_rejected(self, marker):
        """C3 regression: '17 Private cloud:' should NOT pass without seen_ids."""
        m = marker.match("17 Private cloud:")
        assert m and m.group("number") == "17"
        assert not _is_question_header(m, "17 Private cloud:")

    def test_circled_digit_accepted(self, marker):
        m = marker.match("1 Implementation")
        assert m
        assert _is_question_header(m, "\u2460 Implementation")


# ── Full segmentation integration test ──────────────────────────────

class TestSegmentationIntegration:
    """Test segment_normalized_document with synthetic normalized data."""

    def _make_normalized(self, pages_data):
        return {
            "document_id": "test",
            "pages": [
                {
                    "page_number": pn,
                    "elements": [{"type": "text", "text": line} for line in lines],
                }
                for pn, lines in pages_data
            ],
        }

    def test_c4_banner_filtered_and_questions_found(self, tmp_path):
        """C4 scenario: banner lines + question markers without keywords."""
        norm = self._make_normalized([
            (1, [
                "Loading model...",
                "\u2584\u2584 \u2584\u2584",
                "\u2588\u2588 \u2588\u2588",
                "build      : b10424",
                "1) Cluster Computing Implementation",
                "Cluster computing refers to interconnected computers.",
            ]),
            (2, [
                "Loading model...",
                "2) POBHC, Private, Hybrid Cloud",
                "Cloud computing has revolutionized IT.",
            ]),
        ])
        norm_path = tmp_path / "normalized.json"
        norm_path.write_text(json.dumps(norm), encoding="utf-8")
        result = segment_normalized_document(str(norm_path), str(tmp_path / "out"))
        assert result.question_count == 2

        seg = json.loads((tmp_path / "out" / "test" / "segmentation.json").read_text())
        assert seg["questions"][0]["question_id"] == "Q1"
        assert seg["questions"][1]["question_id"] == "Q2"
        assert "Loading model" not in seg["questions"][0]["answer_text"]

    def test_c3_misread_17_stays_in_q2_body(self, tmp_path):
        """C3 scenario: OCR misreads '1)' as '17' inside Q2's body."""
        norm = self._make_normalized([
            (1, [
                "1) How can VIT university implement cluster computing",
                "Cluster computing involves interconnected computers.",
            ]),
            (8, [
                "2) How can VIT leverage cloud computing models",
                "Cloud computing has revolutionized operations.",
            ]),
            (9, [
                "Advantages of cloud computing:",
                "17 Private cloud:",
                "It is dedicated infrastructure for the university.",
            ]),
        ])
        norm_path = tmp_path / "normalized.json"
        norm_path.write_text(json.dumps(norm), encoding="utf-8")
        result = segment_normalized_document(str(norm_path), str(tmp_path / "out"))
        assert result.question_count == 2

        seg = json.loads((tmp_path / "out" / "test" / "segmentation.json").read_text())
        ids = [q["question_id"] for q in seg["questions"]]
        assert "Q17" not in ids
        assert ids == ["Q1", "Q2"]
        assert "Private cloud" in seg["questions"][1]["answer_text"]

    def test_qn_prefix_recognized(self, tmp_path):
        norm = self._make_normalized([
            (1, ["Qn.1. Implementation of cluster computing", "Details."]),
            (5, ["Qn.2. Cloud computing provides scalable solutions", "More details."]),
        ])
        norm_path = tmp_path / "normalized.json"
        norm_path.write_text(json.dumps(norm), encoding="utf-8")
        result = segment_normalized_document(str(norm_path), str(tmp_path / "out"))
        assert result.question_count == 2

    def test_ans_prefix_recognized(self, tmp_path):
        norm = self._make_normalized([
            (1, ["Ans 1. Implementation of cluster computing at VIT", "Details."]),
            (5, ["Ans 2. Cloud computing models for VIT", "More details."]),
        ])
        norm_path = tmp_path / "normalized.json"
        norm_path.write_text(json.dumps(norm), encoding="utf-8")
        result = segment_normalized_document(str(norm_path), str(tmp_path / "out"))
        assert result.question_count == 2

    def test_subparts_merged_into_parent(self, tmp_path):
        norm = self._make_normalized([
            (1, [
                "Q1(a) First part of the answer",
                "Details of first part.",
                "Q1(b) Second part of the answer",
                "Details of second part.",
                "Q1(c) Third part of the answer",
                "Details of third part.",
            ]),
            (3, ["Q2 Cloud computing overview", "Cloud computing explanation."]),
        ])
        norm_path = tmp_path / "normalized.json"
        norm_path.write_text(json.dumps(norm), encoding="utf-8")
        result = segment_normalized_document(str(norm_path), str(tmp_path / "out"))
        assert result.question_count == 2

        seg = json.loads((tmp_path / "out" / "test" / "segmentation.json").read_text())
        assert seg["questions"][0]["question_id"] == "Q1"
        assert "First part" in seg["questions"][0]["answer_text"]
        assert "Second part" in seg["questions"][0]["answer_text"]
        assert "Third part" in seg["questions"][0]["answer_text"]

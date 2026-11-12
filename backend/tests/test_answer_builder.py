"""Unit tests for answer_builder."""

import unittest

from backend.app.answer_builder import build_student_answers


class AnswerBuilderTests(unittest.TestCase):
    def test_builds_text_answer_and_categorizes_typed_regions(self):
        pages = [
            {
                "page_number": 1,
                "elements": [
                    {"type": "text", "text": "Q1 Physics calculation."},
                    {"type": "formula", "text": "E = mc^2"},
                    {"type": "table", "text": "| Mass | Energy |\n| 1kg | 9e16 J |"},
                    {"type": "figure", "text": "Graph showing energy curve"},
                    {"type": "diagram", "text": "Apparatus diagram"},
                ],
            }
        ]
        answers = build_student_answers(pages)
        self.assertEqual(len(answers), 1)
        ans = answers[0]
        self.assertEqual(ans.question_id, "Q1")
        self.assertEqual(ans.pages, [1])
        self.assertEqual(len(ans.formulas), 1)
        self.assertEqual(ans.formulas[0].raw_text, "E = mc^2")
        self.assertEqual(len(ans.tables), 1)
        self.assertEqual(len(ans.charts), 1)
        self.assertEqual(len(ans.diagrams), 1)
        self.assertIn("Physics calculation", ans.text)
        self.assertGreater(ans.ocr_confidence or 0, 0.5)

    def test_handles_empty_or_whitespace_elements(self):
        pages = [
            {
                "page_number": 1,
                "elements": [
                    {"type": "text", "text": "Q1 Answer begins"},
                    {"type": "text", "text": "   "},
                    {"type": "text", "text": ""},
                ],
            }
        ]
        answers = build_student_answers(pages)
        self.assertEqual(len(answers), 1)
        self.assertEqual(answers[0].text, "Q1 Answer begins")

    def test_multi_page_answer_derives_all_pages(self):
        pages = [
            {
                "page_number": 1,
                "elements": [{"type": "text", "text": "Q1 Started on page 1"}],
            },
            {
                "page_number": 2,
                "elements": [{"type": "text", "text": "Continued on page 2"}],
            },
        ]
        answers = build_student_answers(pages)
        self.assertEqual(len(answers), 1)
        self.assertEqual(answers[0].pages, [1, 2])


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from backend.app.segmentation import segment_normalized_document


def text(value):
    return {"type": "text", "label": "text", "text": value, "order": 1, "bbox": None, "polygon": None}


def figure():
    return {"type": "figure", "label": "image", "text": None, "order": 2, "bbox": [1, 2, 3, 4], "polygon": None}


class SegmentationTests(unittest.TestCase):
    def segment(self, pages):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "normalized.json"
            source.write_text(json.dumps({"document_id": "student_001", "pages": pages}), encoding="utf-8")
            result = segment_normalized_document(source, root / "segmentation")
            return json.loads(Path(result.segmentation_path).read_text(encoding="utf-8"))

    def test_numbered_questions(self):
        result = self.segment([{"page_number": 1, "elements": [text("1. First answer"), text("2. Second answer")]}])
        self.assertEqual([question["question_id"] for question in result["questions"]], ["Q1", "Q2"])
        self.assertEqual(result["questions"][0]["answer_text"], "First answer")

    def test_subquestions(self):
        result = self.segment([{"page_number": 1, "elements": [text("Q3(a) Alpha"), text("Q3(b) Beta")]}])
        self.assertEqual([question["question_id"] for question in result["questions"]], ["Q3(a)", "Q3(b)"])

    def test_answer_continues_on_next_page(self):
        result = self.segment([
            {"page_number": 1, "elements": [text("Q1 Starts here")]},
            {"page_number": 2, "elements": [text("and continues here")]},
        ])
        question = result["questions"][0]
        self.assertEqual(question["source_pages"], [1, 2])
        self.assertEqual(question["answer_text"], "Starts here\nand continues here")

    def test_figure_is_attached_to_current_question(self):
        result = self.segment([{"page_number": 1, "elements": [text("Q2 Diagram answer"), figure()]}])
        question = result["questions"][0]
        self.assertEqual(len(question["figures"]), 1)
        self.assertEqual(question["figures"][0]["page_number"], 1)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from backend.app.answer_key_parser import parse_answer_key_file
from backend.app.segmentation import segment_normalized_document


ROOT = Path(__file__).resolve().parents[2]
ANSWER_KEY_FIXTURE = ROOT / "data" / "synthetic" / "answer_key.json"
STUDENT_FIXTURE = ROOT / "data" / "synthetic" / "student_normalized.json"


class SyntheticFixtureTests(unittest.TestCase):
    def test_answer_key_contains_expected_questions_and_explicit_marks(self):
        with tempfile.TemporaryDirectory() as directory:
            result = parse_answer_key_file(ANSWER_KEY_FIXTURE, Path(directory) / "keys")
            payload = json.loads(Path(result.answer_key_path).read_text(encoding="utf-8"))
        self.assertEqual([item["question_id"] for item in payload["questions"]], ["Q1", "Q2", "Q3(a)", "Q3(b)"])
        self.assertEqual([item["maximum_marks"] for item in payload["questions"]], [5, 4, 3, 3])

    def test_student_fixture_segments_to_corresponding_questions(self):
        with tempfile.TemporaryDirectory() as directory:
            result = segment_normalized_document(STUDENT_FIXTURE, Path(directory) / "segments")
            payload = json.loads(Path(result.segmentation_path).read_text(encoding="utf-8"))
        self.assertEqual([item["question_id"] for item in payload["questions"]], ["Q1", "Q2", "Q3(a)", "Q3(b)"])
        self.assertEqual(payload["questions"][2]["source_pages"], [2])


if __name__ == "__main__":
    unittest.main()

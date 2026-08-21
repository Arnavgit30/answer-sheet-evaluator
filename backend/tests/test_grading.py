import json
import tempfile
import unittest
from pathlib import Path

from backend.app.answer_key_parser import parse_answer_key_file
from backend.app.grading import grade_from_files, normalize_text
from backend.app.segmentation import segment_normalized_document


ROOT = Path(__file__).resolve().parents[2]
KEY_FIXTURE = ROOT / "data" / "synthetic" / "answer_key.json"
STUDENT_FIXTURE = ROOT / "data" / "synthetic" / "student_normalized.json"


class BaselineGradingTests(unittest.TestCase):
    def grade_fixture(self, replacements=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            student = json.loads(STUDENT_FIXTURE.read_text(encoding="utf-8"))
            for page in student["pages"]:
                for element in page["elements"]:
                    if replacements and element["text"][:2] in replacements:
                        element["text"] = replacements[element["text"][:2]]
            student_file = root / "student.json"
            student_file.write_text(json.dumps(student), encoding="utf-8")
            segments = segment_normalized_document(student_file, root / "segments")
            key = parse_answer_key_file(KEY_FIXTURE, root / "keys")
            result = grade_from_files(segments.segmentation_path, key.answer_key_path, root / "grading")
            return json.loads(Path(result.grading_path).read_text(encoding="utf-8"))

    def test_near_correct_answer_receives_high_baseline_mark(self):
        result = self.grade_fixture()
        q1 = result["questions"][0]
        self.assertGreater(q1["similarity_score"], 75)
        self.assertGreater(q1["predicted_marks"], 3.5)
        self.assertEqual(q1["grading_method"], "baseline_similarity")

    def test_partially_matching_answer_receives_bounded_partial_mark(self):
        result = self.grade_fixture({"Q2": "Q2 Water changes into vapour."})
        q2 = result["questions"][1]
        self.assertGreater(q2["predicted_marks"], 0)
        self.assertLess(q2["predicted_marks"], q2["maximum_marks"])

    def test_unrelated_answer_has_low_similarity(self):
        result = self.grade_fixture({"Q3": "Q3(a) The moon is made of cheese."})
        q3 = result["questions"][2]
        self.assertLess(q3["similarity_score"], 50)

    def test_missing_answer_is_scored_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = parse_answer_key_file(KEY_FIXTURE, root / "keys")
            student = root / "segments.json"
            student.write_text(json.dumps({"document_id": "student_001", "questions": []}), encoding="utf-8")
            result = grade_from_files(student, key.answer_key_path, root / "grading")
            payload = json.loads(Path(result.grading_path).read_text(encoding="utf-8"))
        self.assertTrue(all(question["predicted_marks"] == 0 for question in payload["questions"]))

    def test_text_normalization_is_deterministic(self):
        self.assertEqual(normalize_text(" Light-energy,  GLUCOSE! "), "light energy glucose")


if __name__ == "__main__":
    unittest.main()

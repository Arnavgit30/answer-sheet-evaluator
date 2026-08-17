import json
import tempfile
import unittest
from pathlib import Path

from backend.app.answer_key_parser import AnswerKeyParseConfig, parse_answer_key_file


class AnswerKeyParserTests(unittest.TestCase):
    def test_parses_text_questions_subquestions_and_configured_marks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "key.txt"
            source.write_text("Q1 First key answer\nQ2(a) Sub answer\ncontinued", encoding="utf-8")
            config = AnswerKeyParseConfig(maximum_marks={"Q1": 5})
            first = parse_answer_key_file(source, root / "out", config)
            second = parse_answer_key_file(source, root / "out", config)
            result = json.loads(Path(first.answer_key_path).read_text(encoding="utf-8"))
        self.assertFalse(first.cached)
        self.assertTrue(second.cached)
        self.assertEqual(result["questions"][0], {"question_id": "Q1", "answer_text": "First key answer", "maximum_marks": 5, "source_pages": [1]})
        self.assertEqual(result["questions"][1]["question_id"], "Q2(a)")
        self.assertEqual(result["questions"][1]["answer_text"], "Sub answer\ncontinued")
        self.assertIsNone(result["questions"][1]["maximum_marks"])

    def test_preserves_explicit_structured_marks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "key.json"
            source.write_text(json.dumps({"questions": [{"question_id": "Q3", "answer_text": "Structured key", "maximum_marks": 10, "source_pages": [2]}]}), encoding="utf-8")
            result = parse_answer_key_file(source, root / "out")
            parsed = json.loads(Path(result.answer_key_path).read_text(encoding="utf-8"))
        self.assertEqual(parsed["questions"][0]["maximum_marks"], 10)
        self.assertEqual(parsed["questions"][0]["source_pages"], [2])


if __name__ == "__main__":
    unittest.main()

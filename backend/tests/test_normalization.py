import tempfile
import unittest
from pathlib import Path

from backend.app.normalization import normalize_raw_ocr


ROOT = Path(__file__).resolve().parents[2]
RAW_FIXTURE = ROOT / "outputs" / "ocr" / "student_001" / "ocr_raw.json"


@unittest.skipUnless(RAW_FIXTURE.is_file(), "Run the Phase D OCR fixture first")
class NormalizationTests(unittest.TestCase):
    def test_normalizes_existing_raw_ocr_fixture_without_changing_it(self):
        before = RAW_FIXTURE.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            first = normalize_raw_ocr(RAW_FIXTURE, directory)
            second = normalize_raw_ocr(RAW_FIXTURE, directory)
            normalized = Path(first.normalized_path).read_text(encoding="utf-8")
        self.assertFalse(first.cached)
        self.assertTrue(second.cached)
        self.assertEqual(RAW_FIXTURE.read_bytes(), before)
        self.assertEqual(first.page_count, 1)
        self.assertIn('"elements"', normalized)
        self.assertIn('"text"', normalized)
        self.assertIn('"bbox"', normalized)
        self.assertIn('"polygon"', normalized)


if __name__ == "__main__":
    unittest.main()

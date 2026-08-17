import tempfile
import unittest
from pathlib import Path

import fitz

from backend.app.dataset_adapter import Document
from backend.app.preprocessing import preprocess_document


class PreprocessingTests(unittest.TestCase):
    def test_copies_image_and_reuses_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "student.jpg"
            source.write_bytes(b"tiny image fixture")
            document = Document("student_001", [str(source)])
            first = preprocess_document(document, root / "processed")
            second = preprocess_document(document, root / "processed")
            self.assertFalse(first.cached)
            self.assertTrue(second.cached)
            self.assertEqual(first.pages[0].page_number, 1)
            self.assertTrue(Path(first.pages[0].processed_file).is_file())

    def test_renders_each_pdf_page(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "student.pdf"
            pdf = fitz.open()
            pdf.new_page().insert_text((72, 72), "page one")
            pdf.new_page().insert_text((72, 72), "page two")
            pdf.save(source)
            pdf.close()
            result = preprocess_document(Document("student_001", [str(source)]), root / "processed")
            self.assertEqual([page.page_number for page in result.pages], [1, 2])
            self.assertTrue(all(Path(page.processed_file).is_file() for page in result.pages))


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from backend.app.dataset_adapter import build_manifest, validate_dataset


class DatasetAdapterTests(unittest.TestCase):
    def test_discovers_flat_scripts_and_page_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data/raw/student_answers/student_001.pdf").parent.mkdir(parents=True)
            (root / "data/raw/student_answers/student_001.pdf").touch()
            (root / "data/raw/student_answers/student_002").mkdir()
            (root / "data/raw/student_answers/student_002/page_001.jpg").touch()
            (root / "data/raw/student_answers/student_002/page_002.png").touch()
            (root / "data/raw/answer_keys").mkdir()
            (root / "data/raw/answer_keys/key_001.pdf").touch()
            config = root / "configs/dataset_config.yaml"
            config.parent.mkdir()
            config.write_text(
                "dataset_name: tiny_demo\nstudent_answers:\n  path: data/raw/student_answers\n"
                "  extensions: [pdf, png, jpg, jpeg]\nanswer_keys:\n  path: data/raw/answer_keys\n"
                "human_marks:\n  available: false\n  path: null\n",
                encoding="utf-8",
            )
            manifest = build_manifest(config)
            self.assertEqual(manifest.dataset_id, "tiny_demo")
            self.assertEqual([item.document_id for item in manifest.student_documents], ["student_001", "student_002"])
            self.assertEqual(len(manifest.student_documents[1].files), 2)
            self.assertEqual(manifest.student_documents[0].answer_key_id, "key_001")
            self.assertTrue(validate_dataset(config)["valid"])


if __name__ == "__main__":
    unittest.main()

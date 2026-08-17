import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.app.ocr_service import LlamaCppOCRService
from backend.app.preprocessing import PreprocessedDocument, ProcessedPage


class OCRServiceTests(unittest.TestCase):
    def test_cli_image_and_writes_normalizer_compatible_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            page_file = root / "page_001.jpg"
            page_file.write_bytes(b"processed fixture")

            document = PreprocessedDocument(
                "student_001",
                [ProcessedPage(1, "source.jpg", str(page_file))],
                cached=False,
            )

            fake_result = type(
                "FakeResult",
                (),
                {
                    "returncode": 0,
                    "stdout": "Q1. First answer\nQ2. Second answer\n",
                    "stderr": "",
                },
            )()

            with patch(
                "backend.app.ocr_service.subprocess.run",
                return_value=fake_result,
            ) as run:
                service = LlamaCppOCRService(
                    timeout_seconds=12,
                    cli_path="llama-cli",
                    model_path="model.gguf",
                    mmproj_path="mmproj.gguf",
                )

                result = service.ocr_document(document, root / "ocr")

            self.assertFalse(result.cached)

            command = run.call_args.args[0]

            self.assertEqual(command[0], "llama-cli")
            self.assertIn("-m", command)
            self.assertIn("model.gguf", command)
            self.assertIn("--mmproj", command)
            self.assertIn("mmproj.gguf", command)
            self.assertIn("--image", command)
            self.assertIn(str(page_file), command)
            self.assertIn("-p", command)
            self.assertIn("OCR:", command)
            self.assertIn("-n", command)
            self.assertIn("512", command)
            self.assertIn("--temp", command)
            self.assertIn("0", command)

            self.assertEqual(
                run.call_args.kwargs["timeout"],
                12,
            )

            raw = json.loads(
                Path(result.raw_ocr_path).read_text(encoding="utf-8")
            )

            blocks = raw["pages"][0]["paddle_output"]["parsing_res_list"]

            self.assertEqual(
                [block["block_content"] for block in blocks],
                ["Q1. First answer", "Q2. Second answer"],
            )

    def test_reuses_raw_ocr_when_processed_page_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            page_file = root / "page_001.jpg"
            page_file.write_bytes(b"processed fixture")

            document = PreprocessedDocument(
                "student_001",
                [ProcessedPage(1, "source.jpg", str(page_file))],
                cached=False,
            )

            fake_result = type(
                "FakeResult",
                (),
                {
                    "returncode": 0,
                    "stdout": "Q1. Answer\n",
                    "stderr": "",
                },
            )()

            with patch(
                "backend.app.ocr_service.subprocess.run",
                return_value=fake_result,
            ) as run:
                service = LlamaCppOCRService()

                first = service.ocr_document(document, root / "ocr")
                second = service.ocr_document(document, root / "ocr")

            self.assertFalse(first.cached)
            self.assertTrue(second.cached)

            # llama-cli should only run once because the second call uses cache.
            self.assertEqual(run.call_count, 1)

    def test_health_check_uses_configured_server(self):
        class FakeHTTPResponse:
            status_code = 200

        with patch(
            "backend.app.ocr_service.requests.get",
            return_value=FakeHTTPResponse(),
        ) as get:
            self.assertTrue(
                LlamaCppOCRService(
                    server_url="http://llama.test:8111"
                ).health_check()
            )

        self.assertEqual(
            get.call_args.args[0],
            "http://llama.test:8111/health",
        )


if __name__ == "__main__":
    unittest.main()
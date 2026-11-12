import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.app.ocr_service import LlamaCppOCRService
from backend.app.preprocessing import PreprocessedDocument, ProcessedPage


class OCRServiceTests(unittest.TestCase):
    def _fake_response(self, text):
        return {"choices": [{"message": {"content": text}}]}

    def test_llama_cpp_transcribes_and_writes_normalizer_compatible_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            page_file = root / "page_001.png"
            page_file.write_bytes(b"PNG_FIXTURE")

            document = PreprocessedDocument(
                "student_001",
                [ProcessedPage(1, "source.png", str(page_file))],
                cached=False,
            )

            mock_llm = MagicMock()
            mock_llm.create_chat_completion.return_value = self._fake_response("Q1. First answer\nQ2. Second answer")

            with patch("backend.app.ocr_service._get_llm_instance", return_value=mock_llm):
                service = LlamaCppOCRService()
                result = service.ocr_document(document, root / "ocr")

            self.assertFalse(result.cached)
            self.assertTrue(mock_llm.create_chat_completion.called)

            call_kwargs = mock_llm.create_chat_completion.call_args.kwargs
            self.assertEqual(call_kwargs["temperature"], 0)
            self.assertEqual(call_kwargs["max_tokens"], 512)

            # Verify output shape is normalizer-compatible.
            raw = json.loads(Path(result.raw_ocr_path).read_text(encoding="utf-8"))
            blocks = raw["pages"][0]["paddle_output"]["parsing_res_list"]
            self.assertEqual(
                [block["block_content"] for block in blocks],
                ["Q1. First answer", "Q2. Second answer"],
            )

    def test_reuses_raw_ocr_when_processed_page_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            page_file = root / "page_001.png"
            page_file.write_bytes(b"PNG_FIXTURE")

            document = PreprocessedDocument(
                "student_001",
                [ProcessedPage(1, "source.png", str(page_file))],
                cached=False,
            )

            mock_llm = MagicMock()
            mock_llm.create_chat_completion.return_value = self._fake_response("Q1. Answer")

            with patch("backend.app.ocr_service._get_llm_instance", return_value=mock_llm):
                service = LlamaCppOCRService()
                first = service.ocr_document(document, root / "ocr")
                second = service.ocr_document(document, root / "ocr")

            self.assertFalse(first.cached)
            self.assertTrue(second.cached)
            # Inference should only fire once; second call uses the cache.
            self.assertEqual(mock_llm.create_chat_completion.call_count, 1)

    def test_health_check_returns_true_when_model_loads(self):
        mock_llm = MagicMock()
        with patch("backend.app.ocr_service._get_llm_instance", return_value=mock_llm):
            self.assertTrue(LlamaCppOCRService().health_check())

    def test_health_check_returns_false_when_model_fails(self):
        with patch("backend.app.ocr_service._get_llm_instance", side_effect=RuntimeError("load failed")):
            self.assertFalse(LlamaCppOCRService().health_check())


if __name__ == "__main__":
    unittest.main()
from paddleocr import PaddleOCRVL

print("Initializing PaddleOCR-VL-1.6...")

pipeline = PaddleOCRVL(
    pipeline_version="v1.6"
)

print("Model initialized.")

output = pipeline.predict("data/raw/test.jpg")

for res in output:
    res.print()
    res.save_to_json(save_path="outputs/ocr_test")
    res.save_to_markdown(save_path="outputs/ocr_test")

print("Done.")
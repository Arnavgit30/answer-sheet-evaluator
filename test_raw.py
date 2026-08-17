import base64
import requests

image_path = r"outputs\processed\C1\page_005.png"

with open(image_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

payload = {
    "prompt": "OCR:\n",
    "image_data": f"data:image/png;base64,{b64}",
    "temperature": 0,
    "n_predict": 512,
}

r = requests.post(
    "http://127.0.0.1:8111/completion",
    json=payload,
    timeout=600,
)

print("HTTP:", r.status_code)
print(r.text)
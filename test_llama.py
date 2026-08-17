import base64
import json
import requests

image_path = r"outputs\processed\C1\page_001.png"

with open(image_path, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

payload = {
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Read this handwritten answer sheet image. Transcribe all visible text exactly. Preserve question numbers and describe any diagrams."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}"
                    }
                }
            ]
        }
    ],
    "temperature": 0,
    "max_tokens": 2048
}

response = requests.post(
    "http://127.0.0.1:8111/v1/chat/completions",
    json=payload,
    timeout=600
)

print("HTTP:", response.status_code)
print(json.dumps(response.json(), indent=2, ensure_ascii=False))
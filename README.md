# Answer Sheet Evaluator

Local solution demo for evaluating handwritten answer sheets with a local llama.cpp PaddleOCR-VL server and a transparent baseline similarity grader.

The OCR client uses `http://127.0.0.1:8111` by default. Set `LLAMA_CPP_SERVER_URL` to use another llama.cpp server URL.

## Start FastAPI

From the repository root:

```cmd
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

## Start the frontend

From `frontend`:

```cmd
npm.cmd install
npm.cmd run dev
```

Open the URL Vite prints (normally `http://127.0.0.1:5173`). The frontend calls FastAPI at `http://127.0.0.1:8000` by default. To use another API URL, copy `.env.example` to `.env` and set `VITE_API_BASE_URL` before starting Vite.

The UI validates the configured local dataset, selects one discovered student document, starts its cached evaluation, then displays baseline grading and optional benchmark results.


-------------------------------------------------------------------

# GradeP — OCR / llama.cpp  README

## Purpose

GradeP is a local handwritten answer-sheet evaluation pipeline.

Target pipeline:

Answer Sheet
→ preprocessing
→ PaddleOCR-VL-1.6
→ raw OCR
→ normalization
→ question/answer segmentation
→ student answer vs answer key
→ transparent baseline similarity
→ UI

## IMPORTANT: current project state

The PaddleOCR-VL-1.6 GGUF model itself works.

The complete six-page OCR fixture has been successfully run with `llama-cli`.

The intended production/scalable architecture is **NOT llama-cli**. It is:

React
→ FastAPI
→ persistent llama-server
→ HTTP `/v1/chat/completions`
→ PaddleOCR-VL-1.6 GGUF
→ OCR JSON
→ downstream pipeline

The production llama-server path is currently blocked by a llama.cpp chat-parser / PEG-native-format error. This README documents that exact problem.

---

## Project paths

Repository root:

```text
C:\Users\Arnav chanchal\Desktop\GradeP\answer-sheet-evaluator
```

Python environment:

```text
.venv\Scripts\python.exe
```

**Always use `.venv\Scripts\python.exe`, not `python`.**

Model directory:

```text
models\paddleocr-vl-gguf
```

Model:

```text
PaddleOCR-VL-1.6-GGUF.gguf
```

Multimodal projector:

```text
PaddleOCR-VL-1.6-GGUF-mmproj.gguf
```

Custom template:

```text
models\paddleocr-vl-gguf\chat_template.jinja
```

Absolute template path:

```text
C:\Users\Arnav chanchal\Desktop\GradeP\answer-sheet-evaluator\models\paddleocr-vl-gguf\chat_template.jinja
```

---

## Python / FastAPI

FastAPI entry point:

```text
backend\app\main.py
```

Start backend:

```cmd
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Development:

```cmd
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

Frontend:

```cmd
cd frontend
npm.cmd install
npm.cmd run dev
```

Normally:

```text
http://127.0.0.1:5173
```

---

# llama.cpp status

## Installed version

```text
version: 0.1.0-dev
build 10424
commit 2bacf9ea5
built with Clang 20.1.8 for Windows x86_64
```

`llama-server -h` showed these relevant options:

```text
--jinja
--chat-template
--chat-template-file
--skip-chat-parsing
```

## Desired server

```text
http://127.0.0.1:8111
```

The original OCR client was intended to use:

```text
POST http://127.0.0.1:8111/v1/chat/completions
```

with:

```json
{
  "temperature": 0,
  "max_tokens": 512
}
```

Do not arbitrarily increase the token limit.

---

# Why llama-server is currently NOT working reliably

The model loads successfully in llama-server.

The server can sometimes return HTTP 200 and useful OCR.

However, the recurring failure is:

```text
HTTP 500
```

with:

```json
{
  "error": {
    "code": 500,
    "message": "The model produced output that does not match the expected peg-native format",
    "type": "server_error"
  }
}
```

The server log shows:

```text
W common_chat_peg_parse: unparsed peg-native output: <OCR text>
```

then:

```text
W srv stop: cancel task
```

and:

```text
W srv operator(): got exception:
{"error":{"code":500,"message":"The model produced output that does not match the expected peg-native format","type":"server_error"}}
```

This is the main unresolved blocker.

**Do not assume the model is broken. It is not.**

The evidence is that the same model successfully performs OCR through llama-cli.

---

# Custom chat template

The template downloaded for PaddleOCR-VL-1.6 GGUF is:

```jinja
{%- if not add_generation_prompt is defined -%}
    {%- set add_generation_prompt = true -%}
{%- endif -%}
{%- if not cls_token is defined -%}
    {%- set cls_token = "<|begin_of_sentence|>" -%}
{%- endif -%}
{%- if not eos_token is defined -%}
    {%- set eos_token = "</s>" -%}
{%- endif -%}
{{- cls_token -}}
{%- for message in messages -%}
    {%- if message["role"] == "user" -%}
        {{- "User: " -}}
      {%- if message["content"] is string -%}
        {{- message["content"] }}
      {%- else -%}
        {%- for content in message["content"] -%}
            {%- if content["type"] == "image" -%}
                {{ "<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>" }}
            {%- endif -%}
        {%- endfor -%}
        {%- for content in message["content"] -%}
            {%- if content["type"] == "text" -%}
                {{ content["text"] }}
            {%- endif -%}
        {%- endfor -%}
      {%- endif -%}
        {{ "\n" -}}
    {%- elif message["role"] == "assistant" -%}
        {{- "Assistant:\n" -}}
      {%- if message["content"] is string -%}
        {{- message["content"] }}
      {%- else -%}
        {%- for content in message["content"] -%}
            {%- if content["type"] == "text" -%}
                {{ content["text"] }}
            {%- endif -%}
        {%- endfor -%}
      {%- endif -%}
        {{ eos_token -}}
    {%- elif message["role"] == "system" -%}
      {%- if message["content"] is string -%}
        {{- message["content"] }}
      {%- else -%}
        {%- for content in message["content"] -%}
            {%- if content["type"] == "text" -%}
                {{ content["text"] + "\n" }}
            {%- endif -%}
        {%- endfor -%}
      {%- endif -%}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- "Assistant:\n" -}}
{%- endif -%}
```

Several relative-path mistakes happened during setup. Use the absolute template path if necessary.

---

# Server experiments already performed

Configurations using combinations of:

```text
--jinja
--chat-template-file
--skip-chat-parsing
```

were tried.

Some configurations successfully loaded the model.

However, the HTTP 500 PEG-native parsing problem still occurred.

Therefore:

**The exact reliable llama-server invocation/configuration has NOT been solved.**

Do not claim that it has.

A raw/content-only endpoint experiment returned HTTP 200 but unusable repeated output resembling:

```text
ìng<fcel>谪书四书<fcel>《诗经》卷首...
```

with 512 generated tokens.

This proves that simply bypassing the chat parser is not automatically the solution. Correct multimodal image/template handling is still required.

---

# llama-cli diagnostic path

`llama-cli` successfully proves that PaddleOCR-VL itself works.

Example:

```cmd
llama-cli -m "models\paddleocr-vl-gguf\PaddleOCR-VL-1.6-GGUF.gguf" --mmproj "models\paddleocr-vl-gguf\PaddleOCR-VL-1.6-GGUF-mmproj.gguf" --image "outputs\processed\C1\page_001.png" -p "OCR:" -n 512 --temp 0
```

It produces useful handwritten OCR.

Page 1 produced content including:

```text
Digital Assignment - 1:
Implementation of Cluster computing in ViT...
Types of cluster systems:
High-performance Computing (HPC) clusters...
Use cases in ViT...
```

Page 5 also produced useful OCR containing handwritten public/private/hybrid cloud notes and a hand-drawn diagram.

Therefore:

**PaddleOCR-VL-1.6 + GGUF + mmproj + image input works.**

---

# Why llama-cli is not the final architecture

With CLI, each page can cause a separate process/model load:

```text
page 1 → start llama-cli → load model → OCR → exit
page 2 → start llama-cli → load model → OCR → exit
...
```

That is slow and unsuitable as the intended scalable backend.

The desired architecture keeps the model loaded:

```text
FastAPI
→ persistent llama-server
→ HTTP requests for pages
```

Therefore llama-cli should remain a diagnostic/fallback path only.

---

# Full CLI OCR fixture

Command:

```cmd
.venv\Scripts\python.exe -m backend.scripts.ocr_fixture
```

Successful result:

```json
{
  "document_id": "C1",
  "raw_ocr_path": "C:\Users\Arnav chanchal\Desktop\GradeP\answer-sheet-evaluator\outputs\ocr\C1\ocr_raw.json",
  "page_count": 6,
  "cached": false
}
```

Current OCR raw file:

```text
outputs\ocr\C1\ocr_raw.json
```

Observed size:

```text
70,817 bytes
```

So the full six-page OCR fixture can run with CLI.

---

# Current OCR service situation

During server debugging, `backend/app/ocr_service.py` was temporarily changed to use `subprocess.run()` and `llama-cli`.

The desired final implementation should return to:

```text
requests.post()
```

against:

```text
/v1/chat/completions
```

The original intended request contains:

- user text prompt
- base64 image
- `temperature: 0`
- `max_tokens: 512`

Prompt:

```text
Read this handwritten answer sheet image. Transcribe all visible text exactly. Preserve question numbers and describe any diagrams.
```

The service converts returned transcription to the project's Paddle-compatible shape:

```json
{
  "parsing_res_list": [
    {
      "block_label": "text",
      "block_content": "...",
      "block_order": 0,
      "block_id": 0,
      "block_bbox": null,
      "block_polygon_points": null
    }
  ]
}
```

---

# Unit tests

The OCR tests originally expected HTTP requests.

Because the implementation was temporarily switched to CLI, the tests were temporarily changed to mock subprocess.

The temporary CLI-focused tests eventually passed:

```text
Ran 3 tests in 0.031s

OK
```

Once llama-server HTTP is fixed, tests should be restored to mock:

```python
requests.post
```

and verify:

```text
/v1/chat/completions
temperature = 0
max_tokens = 512
base64 image
```

Do not leave the CLI implementation as the final backend merely because those temporary tests pass.

---

# Downstream outputs

The project currently contains:

```text
outputs\ocr\C1\ocr_raw.json
outputs\normalized\C1\normalized.json
outputs\evaluations\*.json
```

and answer-key normalized outputs such as:

```text
outputs\answer_keys\normalized\C2\normalized.json
outputs\answer_keys\normalized\key_001\normalized.json
```

Thus the pipeline is generating downstream artifacts.

---

# Current UI behavior

The React UI works and can process the selected C1 student sheet.

The UI eventually displayed evaluation results such as:

```text
Q1
Similarity: 0.0%

Student answer
No answer available.
```

and similarly for Q2, Q3 and Q4.

The reference answers are displayed correctly.

This does NOT mean the OCR model failed.

The current problem is downstream and/or contaminated input.

---

# Important downstream diagnosis

The actual student sheet contains handwritten/circled question numbers such as:

```text
①
②
```

OCR can corrupt these characters.

For example, raw OCR has contained malformed representations such as:

```text
â‘¡
```

instead of a clean circled digit.

If the segmentation code expects only:

```text
Q1
Q2
Question 1
Question 2
```

then the student's answers may never be mapped into Q1/Q2/Q3/Q4.

That would explain:

```text
Student answer: No answer available
Similarity: 0%
```

even though OCR text exists.

**However, this is a hypothesis, not yet proven.**

Before changing segmentation, inspect:

```text
outputs\normalized\C1\normalized.json
```

and determine whether student answers actually exist there.

---

# CLI stdout contamination

The CLI can print non-OCR artifacts:

```text
Loading model...
build ...
model ...
ftype ...
modalities ...
Loaded media from ...
> OCR:
Exiting...
```

Some of these artifacts entered raw OCR output.

For example, raw output has included:

```text
Loading model...
...
Exiting...
```

Therefore CLI output cleaning is needed if CLI is used for the demo.

But this is NOT the main production blocker.

The main production blocker remains llama-server's PEG-native parsing.

---

# Page 5 evidence

Page 5 contains:

- a hand-drawn cloud-computing diagram
- public cloud
- VIT learning platforms
- student/faculty figures
- benefits
- use cases
- private cloud
- hybrid cloud

The model can read this page through llama-cli.

Therefore:

**Do not assume the HTTP 500 is simply caused by a complex freestyle figure.**

The PEG-native error has occurred on other content as well.

---

# Current architecture

## Working diagnostic/demo path

```text
React
 ↓
FastAPI
 ↓
llama-cli
 ↓
PaddleOCR-VL-1.6 GGUF
 ↓
OCR JSON
 ↓
normalization
 ↓
evaluation
 ↓
React
```

## Intended scalable path

```text
React
 ↓
FastAPI
 ↓
persistent llama-server
 ↓
HTTP /v1/chat/completions
 ↓
PaddleOCR-VL-1.6 GGUF
 ↓
OCR JSON
 ↓
normalization
 ↓
question/answer segmentation
 ↓
evaluation
 ↓
React
```

---

# Current known facts

1. PaddleOCR-VL-1.6 GGUF loads.
2. The mmproj loads.
3. Handwritten pages can be OCR'd.
4. The six-page C1 CLI fixture works.
5. FastAPI entry point is `backend.app.main:app`.
6. React frontend works.
7. Raw OCR JSON is generated.
8. Normalized and evaluation JSON files are generated.
9. Current UI can display evaluation results.
10. Current CLI OCR path is not the scalable production architecture.
11. llama-server HTTP is the intended architecture.
12. llama-server repeatedly fails with:
   `The model produced output that does not match the expected peg-native format`.
13. Server logs show:
   `common_chat_peg_parse: unparsed peg-native output`.
14. `--skip-chat-parsing` was investigated but did not yet produce a proven reliable HTTP OCR path.
15. A raw HTTP/content-only test returned HTTP 200 but garbage/repeated tokens.
16. Therefore the exact llama-server configuration is still unresolved.
17. Current UI `No answer available` results should not be treated as valid grading/benchmark results.
18. The next debugging step should inspect intermediate normalized output before changing downstream segmentation.

---

# Recommended next steps

Because development/Codex budget is limited, debug one layer at a time.

## Step 1 — prove llama-server HTTP on one image

Start llama-server.

Use one standalone Python HTTP test against:

```text
outputs\processed\C1\page_001.png
```

Confirm:

```text
HTTP 200
```

and inspect returned content.

Then repeat only for:

```text
outputs\processed\C1\page_005.png
```

Do not run all six pages until both work.

## Step 2 — solve PEG-native parsing

Investigate the exact interaction between:

```text
--jinja
--chat-template-file
--skip-chat-parsing
```

and the model's expected output/template.

The error to solve is:

```text
The model produced output that does not match the expected peg-native format
```

## Step 3 — restore HTTP OCR service

Once HTTP OCR is reliable, change `_ocr_page()` back to `requests.post()`.

## Step 4 — restore HTTP-focused tests

Mock `requests.post()` again.

## Step 5 — run C1 through backend

Verify raw OCR JSON is clean.

## Step 6 — inspect normalization

Check:

```text
outputs\normalized\C1\normalized.json
```

Verify whether student answers exist.

## Step 7 — only then fix segmentation

If OCR has answers but normalized output loses them, investigate question-number normalization/segmentation.

## Step 8 — only then trust evaluation/UI

The current baseline similarity result is only meaningful after student-answer mapping is confirmed.

---

# What NOT to do

Do not:

- permanently switch to llama-cli
- claim llama-server is solved
- assume the model is broken
- assume diagrams are the root cause
- increase max_tokens arbitrarily
- repeatedly rewrite the OCR service without isolated server tests
- change segmentation before inspecting normalized output
- treat the current 0% UI result as a benchmark
- assume `--skip-chat-parsing` alone fixes the server
- use global `python`
- spend remaining Codex budget on broad refactors

---

# Fresh-chat instruction

When starting a new chat, give this README to the assistant and say:

> Read this README as the authoritative current GradeP project state. Do not assume anything that is not supported by it. The PaddleOCR-VL model is already proven to work through llama-cli. The goal now is to solve the llama.cpp llama-server HTTP integration so the model can be used persistently and scalably. Debug the server path first, then restore the HTTP OCR service, and only after that debug downstream normalization/segmentation. Make minimal, evidence-driven changes because development/Codex budget is limited.

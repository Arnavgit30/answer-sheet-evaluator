# GradePilot — Answer Sheet Evaluator

A local-first handwritten answer-sheet OCR and AI grading pipeline built with **FastAPI**, **React (Vite + Tailwind)**, **llama-cpp-python (`PaddleOCR-VL-1.6-GGUF`)**, and **OpenAI-compatible LLM grading (`qwen/qwen3.8-27b` on Groq by default)**.

It supports both **direct UI file uploads** (one student answer sheet + one answer key per run) and **pre-configured local dataset manifests**, with SHA-256 caching across pipeline stages and **human-in-the-loop review gates** for low-confidence OCR and grading items.

---

## End-to-End Pipeline Architecture

```text
React UI (File Upload or Dataset Config)
  │
  ▼  POST /api/v1/uploads  or  POST /api/v1/datasets/validate
FastAPI (backend/app/api.py -> EvaluationService)
  │
  ├─► 1. Preprocessing (backend/app/preprocessing.py)
  │      Renders PDFs to PNGs via PyMuPDF (2x scale) or copies PNG/JPG images
  │      -> outputs/processed/<doc_id>/page_NNN.png (cached via preprocess.json)
  │
  ├─► 2. In-Process Multimodal OCR (backend/app/ocr_service.py)
  │      Lazy-loads PaddleOCR-VL-1.6-GGUF + mmproj once in-process via llama-cpp-python
  │      (Llava15ChatHandler, n_ctx=4096) -> outputs/ocr/<doc_id>/ocr_raw.json
  │
  ├─► 3. Normalization & Segmentation
  │      normalization.py -> question_detector.py -> region_associator.py
  │      -> segmentation.py -> answer_builder.py
  │      Filters diagnostic/banner noise, detects standard markers (Q1, Ans 1, 1(a))
  │      and Unicode circled/parenthesized/dotted digits (①–⑳, ⑴–⒇, ⒈–⒑),
  │      merges multi-page answers, and extracts formulas, tables, charts, diagrams
  │
  ├─► 4. Answer Key Parsing (backend/app/answer_key_parser.py)
  │      Supports PDF/image keys (via OCR + normalization) or .txt / .json keys
  │      Preserves optional rubric / marking_scheme and key_concepts (default 5.0 marks)
  │      -> outputs/answer_keys/
  │
  ├─► 5. Semantic LLM Grading (backend/app/vlm_evaluator.py, backend/app/grading.py)
  │      Grades extracted text, formulas, tables, charts, and diagrams against reference
  │      answers, rubrics, and key concepts (no silent similarity fallback)
  │      Blank answers -> 0.0 marks + human_review_required=True
  │      Missing API key -> status="pending" + human_review_required=True
  │      API/validation error -> status="failed" + human_review_required=True
  │      Lexical similarity (rapidfuzz) is stored only as passive metadata
  │
  └─► 6. Review Queues & Benchmarking
         ocr_review.py (flags heuristic OCR confidence < 0.7)
         grading_review.py (flags human_review_required, pending/failed, confidence < 0.7)
         benchmarking.py (MAE, RMSE, Pearson, Spearman, within-1-mark if human marks exist)
```

> **Note on `llama.cpp` Integration:** The older external `llama-server` HTTP path (which suffered from PEG-native chat-parser HTTP 500 errors) and per-page `llama-cli` subprocess calls have been replaced by **`LlamaCppOCRService`** running **`llama-cpp-python` in-process**. The model loads lazily once per backend process and stays warm across pages and requests.

---

## Repository Structure

```text
answer-sheet-evaluator/
├── backend/
│   ├── app/
│   │   ├── main.py                # FastAPI app & CORS setup
│   │   ├── api.py                 # /api/v1 HTTP endpoints & single-run evaluation lock
│   │   ├── api_models.py          # Pydantic request/response schemas
│   │   ├── evaluation_service.py  # End-to-end pipeline orchestration & review updates
│   │   ├── evaluation_store.py    # JSON persistence in outputs/evaluations/
│   │   ├── uploads.py             # Validates & saves uploads under data/uploads/
│   │   ├── dataset_adapter.py     # Dataset config & manifest validation
│   │   ├── preprocessing.py       # PyMuPDF PDF-to-PNG rendering & image caching
│   │   ├── ocr_service.py         # In-process llama-cpp-python PaddleOCR-VL service
│   │   ├── normalization.py       # OCR cleanup & block normalization
│   │   ├── question_detector.py   # Question header & Unicode marker detection
│   │   ├── region_associator.py   # Multi-page question-to-region grouping
│   │   ├── segmentation.py        # Question segmentation artifacts
│   │   ├── answer_builder.py      # Builds StudentAnswer (text, formulas, tables, diagrams)
│   │   ├── answer_key_parser.py   # Parses PDF/image/TXT/JSON answer keys
│   │   ├── vlm_evaluator.py       # Primary OpenAI-compatible LLM grader (Groq/Qwen default)
│   │   ├── grading.py             # LLMGrader + passive similarity helper
│   │   ├── ocr_review.py          # OCR confidence scoring & review queue
│   │   ├── grading_review.py      # Grading confidence scoring & override queue
│   │   ├── benchmarking.py        # Ground-truth benchmark metrics
│   │   └── models.py              # Core dataclasses
│   └── tests/                     # Pytest suite (unit, API, segmentation, grading, reviews)
├── frontend/
│   ├── src/
│   │   └── App.jsx                # React UI: upload/dataset workflows, results & review tabs
│   ├── .env.example               # VITE_API_BASE_URL configuration
│   └── package.json
├── configs/
│   └── dataset_config.yaml        # Default local dataset manifest config
├── data/
│   ├── raw/                       # Local sample/raw student sheets & answer keys
│   ├── synthetic/                 # Synthetic test fixtures
│   └── uploads/                   # Ephemeral UI uploads (git-ignored)
├── models/
│   └── paddleocr-vl-gguf/         # PaddleOCR-VL-1.6-GGUF model & mmproj weights (git-ignored)
├── outputs/                       # Cached artifacts (processed, ocr, normalized, evaluations)
├── .env.example                   # Backend LLM grader env template
└── requirements.txt               # Core Python dependencies
```

---

## Prerequisites & Setup

### 1. Python Environment (`.venv`)

Always run backend commands with the project virtual environment (`.venv`), not global `python`, so `PyMuPDF` and `llama-cpp-python` resolve cleanly:

```cmd
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Local OCR also requires `llama-cpp-python` in the same `.venv`:

```cmd
.\.venv\Scripts\python.exe -m pip install llama-cpp-python
```

### 2. Local OCR Model Files

Place the PaddleOCR-VL GGUF weights under `models\paddleocr-vl-gguf\`:

- `models\paddleocr-vl-gguf\PaddleOCR-VL-1.6-GGUF.gguf`
- `models\paddleocr-vl-gguf\PaddleOCR-VL-1.6-GGUF-mmproj.gguf`

You can override these default paths with `LLAMA_CPP_MODEL_PATH` and `LLAMA_CPP_MMPROJ_PATH`.

### 3. LLM Grader Environment Variables (`.env`)

Copy `.env.example` to `.env` in the repository root (`VLMEvaluator` loads `.env` or `backend/.env` automatically if environment variables are not already set):

```env
VLM_API_KEY=your_groq_api_key_here
VLM_BASE_URL=https://api.groq.com/openai/v1
VLM_MODEL=qwen/qwen3.8-27b
```

- `VLM_API_KEY` (or `GROQ_API_KEY`): API key for grading. If omitted, questions are marked `status="pending"` with `marks_awarded=null` and routed to **Grading Review Gate**.
- `VLM_BASE_URL`: Any OpenAI-compatible chat-completions endpoint (`https://api.groq.com/openai/v1` by default).
- `VLM_MODEL`: Model name (`qwen/qwen3.8-27b` by default).

---

## Running the Application

### Start the FastAPI Backend

From the repository root:

```cmd
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

For development with auto-reload:

```cmd
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

### Start the React Frontend

From `frontend/`:

```cmd
npm.cmd install
npm.cmd run dev
```

Open the URL printed by Vite (typically `http://127.0.0.1:5173`).  
To point the UI at a non-default backend URL, copy `frontend\.env.example` to `frontend\.env` and edit `VITE_API_BASE_URL`.

---

## Using the UI

### Workflow A — Direct File Upload (Single Student per Run)
1. Under **Student answer sheet**, choose a `.pdf` or one or more `.png` / `.jpg` / `.jpeg` page images (max `25 MB` per file).
2. Under **Answer key**, choose a `.pdf`, page image(s), or a single `.txt` / `.json` file (max `25 MB` per file).
3. Click **Upload & Evaluate**.
   - Files are content-hashed and saved under `data/uploads/stu_<sha12>_key_<sha12>/` with a generated `configs/dataset_config.json`.
   - Re-uploading identical files reuses cached preprocessing, OCR, normalization, segmentation, and answer-key artifacts.

### Workflow B — Saved Dataset Config
1. Keep or edit the **Dataset configuration path** (defaults to `configs/dataset_config.yaml`, pointing to `data/raw/student_answers` and `data/raw/answer_keys`).
2. Click **Validate dataset**, pick a student sheet from the dropdown, and click **Start Full Pipeline**.

### Review & Inspection Tabs
- **Evaluated Questions**: Total predicted score, per-question marks (`marks_awarded`), AI reasoning, present/missing concepts, errors, extracted formulas/tables/diagrams, side-by-side reference answer, passive lexical similarity score, and an **Original Exam Page** modal viewer.
- **OCR Review Gate**: Displays answers whose heuristic OCR confidence is below `0.7` so a reviewer can confirm or correct the extracted text.
- **Grading Review Gate**: Displays blank, `pending`, `failed`, or low-confidence (`< 0.7`) evaluations so a teacher can **confirm** AI marks or **override** marks with notes (automatically updating question scores and evaluation totals).

---

## REST API Reference (`/api/v1`)

| Method | Endpoint | Purpose |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Health check |
| `POST` | `/api/v1/uploads` | Upload `student_files` + `answer_key_files`; writes dataset dir & returns `config_path` + `student_document_id` |
| `POST` | `/api/v1/datasets/validate` | Validate a dataset config YAML/JSON and inspect discovered student docs & answer keys |
| `POST` | `/api/v1/evaluations` | Run the full pipeline for a config + `student_document_id` (`409 Conflict` if another run holds `_EVALUATION_LOCK`) |
| `GET` | `/api/v1/evaluations/{evaluation_id}` | Poll evaluation status (`queued`, `running`, `completed`, `failed`) |
| `GET` | `/api/v1/evaluations/{evaluation_id}/results` | Fetch completed grading summary and benchmark metrics |
| `GET` | `/api/v1/evaluations/{evaluation_id}/questions` | Fetch per-question student answers, reference answers, and evaluations |
| `GET` | `/api/v1/evaluations/{evaluation_id}/questions/{question_id}` | Fetch detailed view for a single question |
| `GET` | `/api/v1/evaluations/{evaluation_id}/pages/{page_number}` | Serve processed exam page PNG/JPEG for UI inspection |
| `GET` / `POST` | `/api/v1/evaluations/{evaluation_id}/ocr-reviews` (`/{review_id}`) | List OCR reviews or submit human confirmation/correction |
| `GET` / `POST` | `/api/v1/evaluations/{evaluation_id}/grading-reviews` (`/{review_id}`) | List grading reviews or submit teacher confirmation/override |

---

## Running Tests

From the repository root:

```cmd
.\.venv\Scripts\python.exe -m pytest backend/tests -q
```

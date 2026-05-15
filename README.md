# Document Understanding

> **OCR → Retrieval → Grounded Drafting → PDF Export**
> A GPU-accelerated pipeline that turns scanned PDFs and images into structured markdown, retrieves relevant evidence, and generates operator-quality legal/document drafts — with a built-in feedback loop that learns from your edits.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running the Project](#running-the-project)
- [API Reference](#api-reference)
- [Draft Types](#draft-types)
- [Docker](#docker)
- [Assumptions & Tradeoffs](#assumptions--tradeoffs)

---

## Overview

**Document Understanding** is an end-to-end document intelligence system built around [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR). It accepts scanned PDFs or images, converts them to clean markdown, chunks and indexes the content for semantic retrieval, and generates grounded drafts in multiple formats using Groq-hosted LLMs. An operator feedback loop extracts reusable editing rules from human corrections, improving future drafts automatically.

---

## Features

- **High-quality OCR** via DeepSeek-OCR (CUDA-accelerated)
- **Semantic retrieval** with local ChromaDB and sentence-transformers
- **Grounded drafting** in 5 document formats, powered by Groq (LLaMA 3.3 70B)
- **Feedback loop** that mines operator edits into reusable rules
- **PDF export** rendered from markdown via ReportLab
- **FastAPI backend** with clean REST endpoints
- **Streamlit UI** for interactive upload, review, and export
- **CLI** for batch or single-file pipeline runs
- **Docker** support with optional GPU passthrough

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                              │
│              PDF / Image / Directory of files                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                       OCR PIPELINE                              │
│                      deepseek.py                                │
│                                                                 │
│  PDF → page images (pdf2image) → DeepSeek-OCR (CUDA)           │
│       → clean markdown extraction → structured JSON chunks      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
               ┌────────────┴────────────┐
               │                         │
               ▼                         ▼
    ┌──────────────────┐      ┌────────────────────────┐
    │  OCR Markdown    │      │  Chunked JSON           │
    │  (.md file)      │      │  (.json file)           │
    └──────────────────┘      └──────────┬─────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RETRIEVAL LAYER                              │
│                   pipeline/retrieval.py                         │
│                                                                 │
│  Chunked JSON → sentence-transformers embeddings                │
│              → ChromaDB (local vector store)                    │
│              → top-k passage retrieval by draft-type query      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DRAFTING LAYER                               │
│                   pipeline/drafting.py                          │
│                                                                 │
│  Retrieved passages + feedback rules → Groq LLM prompt          │
│                                     → grounded draft markdown   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
               ┌────────────┴────────────┐
               │                         │
               ▼                         ▼
    ┌──────────────────┐      ┌────────────────────────┐
    │  Draft Markdown  │      │   Draft JSON            │
    │  (.draft.md)     │      │   (evidence + metadata) │
    └──────────────────┘      └────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FEEDBACK LOOP                                │
│                   pipeline/feedback.py                          │
│                                                                 │
│  Operator edits draft → diff captured → LLM mines edit rules    │
│                      → rules saved per draft-type               │
│                      → injected into future drafting prompts    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     EXPORT LAYER                                │
│                  pipeline/pdf_render.py                         │
│                                                                 │
│  Edited draft markdown → ReportLab → PDF bytes → download       │
└─────────────────────────────────────────────────────────────────┘


  ┌───────────────────┐          ┌──────────────────────┐
  │   FastAPI (api.py)│◄────────►│  Streamlit UI         │
  │   :8000           │  HTTP    │  (streamlit_app.py)   │
  └───────────────────┘          │  :8501                │
           ▲                     └──────────────────────┘
           │ CLI
  ┌────────┴──────────┐
  │   main.py         │
  │   (argparse CLI)  │
  └───────────────────┘
```

**Data flow summary:**

1. A PDF or image enters the system via the UI, API, or CLI
2. DeepSeek-OCR converts pages to clean markdown and a chunked JSON index
3. ChromaDB embeds and stores the chunks; the top-k most relevant passages are retrieved for the chosen draft type
4. Groq (LLaMA 3.3 70B) generates a grounded draft, citing only retrieved evidence
5. The operator reviews and edits the draft in the UI; edits are submitted as feedback
6. The feedback pipeline mines reusable rules and persists them for future runs
7. The final draft is exported as PDF or downloaded as markdown

---

## Project Structure

```
document-understanding/
├── deepseek.py              # DeepSeek-OCR model loading and inference
├── api.py                   # FastAPI backend (OCR, drafting, feedback, PDF)
├── streamlit_app.py         # Streamlit browser UI
├── main.py                  # CLI entrypoint
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container image definition
├── docker-compose.yml       # Base compose (CPU)
├── docker-compose.gpu.yml   # GPU overlay for NVIDIA passthrough
│
├── pipeline/
│   ├── ocr.py               # OCR orchestration (wraps deepseek.py)
│   ├── retrieval.py         # ChromaDB indexing and passage retrieval
│   ├── drafting.py          # Groq-powered grounded draft generation
│   ├── feedback.py          # Operator edit capture and rule mining
│   └── pdf_render.py        # Markdown → PDF via ReportLab
│
├── data/
│   ├── input/               # Drop PDFs and images here
│   ├── output/              # OCR markdown, JSON chunks, drafts
│   └── chroma_db/           # Persisted ChromaDB vector store
│
└── scripts/
    └── test_draft_without_ocr.py   # Test drafting from existing JSON (no GPU needed)
```

---

## Requirements

### Hardware

- NVIDIA GPU with CUDA support (required for DeepSeek-OCR inference)
- Minimum 16 GB VRAM recommended for comfortable throughput

### System dependencies

- Python 3.10+
- Poppler (for PDF-to-image conversion)
- CUDA toolkit (matching your PyTorch build)

### External services

- [Groq API key](https://console.groq.com/) — used for LLM-powered drafting and feedback rule extraction

---

## Setup

### 1. Install system dependencies

```bash
sudo apt-get update
sudo apt-get install -y poppler-utils
```

### 2. Create a Python virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Install Flash Attention for faster OCR

Only install this if your CUDA stack supports it:

```bash
pip install flash-attn==2.7.3 --no-build-isolation
```

---

## Configuration

Create a `.env` file in the project root:

```bash
GROQ_API_KEY=your-groq-api-key-here
```

Or export it for a single session:

```bash
export GROQ_API_KEY="your-groq-api-key-here"
```

The key is loaded automatically by `python-dotenv` when the API or CLI starts.

---

## Running the Project

### FastAPI backend

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Interactive API docs available at `http://localhost:8000/docs`

### Streamlit UI

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501`, set the API URL to `http://localhost:8000`, upload a PDF or image, select a draft type, and click **Run OCR + Draft**.

### CLI pipeline

**Single file:**

```bash
python main.py --input data/input/document.pdf --draft-type case_fact_summary
```

**Entire directory:**

```bash
python main.py --input data/input/ --draft-type notice_related_summary
```

**All CLI options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `data/input` | Input file or directory |
| `--output-dir` | `data/output` | Output directory for OCR and drafts |
| `--draft-type` | *(required)* | One of the five draft types (see below) |
| `--top-k` | `8` | Number of evidence passages to retrieve |
| `--dpi` | `200` | PDF-to-image DPI (higher = better quality, slower) |
| `--base-size` | `1024` | OCR model base size parameter |
| `--image-size` | `640` | OCR model image size parameter |
| `--model` | `llama-3.3-70b-versatile` | Groq model name |
| `--output-draft` | auto | Custom output path for draft markdown |
| `--output-draft-json` | auto | Custom output path for draft JSON |

### Test drafting without a GPU

If you want to test the retrieval and drafting stages without running OCR, provide an existing OCR JSON file:

```bash
python scripts/test_draft_without_ocr.py \
  --json-path data/output/document.json \
  --draft-type case_fact_summary
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/inputs` | List files in `data/input/` |
| `POST` | `/process` | Run full pipeline on uploaded file or existing input |
| `POST` | `/feedback` | Submit operator edits to extract reusable rules |
| `POST` | `/generate-pdf` | Render a markdown draft to PDF |

### `POST /process` — request fields

| Field | Type | Description |
|-------|------|-------------|
| `file` | file upload | PDF or image to process |
| `input_name` | string | Filename from `data/input/` (alternative to upload) |
| `draft_type` | string | One of the five draft type keys |
| `top_k` | int | Evidence passages to retrieve (default 8) |
| `dpi` | int | OCR rendering DPI (default 200) |
| `model` | string | Groq model name |

### `POST /process` — response fields

| Field | Description |
|-------|-------------|
| `ocr_markdown` | Full OCR output as markdown |
| `draft_markdown` | Generated draft markdown |
| `evidence_passages` | All retrieved passages with scores |
| `evidence_used` | Passages cited in the draft |
| `ocr_md_path` | Server path to OCR markdown file |
| `draft_md_path` | Server path to draft markdown file |

---

## Draft Types

| Key | Description |
|-----|-------------|
| `title_review_summary` | Summary of title review findings |
| `case_fact_summary` | Structured summary of case facts |
| `notice_related_summary` | Summary of notices and related documents |
| `document_checklist` | Checklist of required or present documents |
| `first_pass_internal_memo` | Internal memo from a first-pass document review |

---

## Docker

### Build and run (CPU)

```bash
docker compose up --build
```

- API: `http://localhost:8000`
- UI: `http://localhost:8501`

The `data/` directory is mounted into the container so inputs and outputs persist across restarts.

### Build and run (NVIDIA GPU)

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

### Clone repo without git (optional)

```bash
docker run --rm -v "$PWD":/work -w /work alpine/git clone <repo-url> document-understanding
cd document-understanding
```

---

## Assumptions & Tradeoffs

- **Groq dependency** — draft quality and availability depend on your network connection and Groq uptime. Swap the model string for any Groq-supported model.
- **GPU requirement** — DeepSeek-OCR is GPU-heavy. CPU-only runs are possible but significantly slower, especially for multi-page PDFs.
- **OCR fidelity** — retrieval and drafting quality are upstream of OCR accuracy. Noisy or low-DPI scans reduce evidence recall.
- **Local ChromaDB** — convenient for single-user local use, but not designed for multi-tenant or access-controlled deployments.
- **No auth or rate limiting** — the FastAPI backend has no authentication, user management, or rate limiting built in. Add a reverse proxy (e.g. Nginx + OAuth2 Proxy) for production use.
- **Feedback rules are global** — rules extracted from operator edits are stored per draft type, not per user or document. All future runs of a given draft type inherit accumulated rules.
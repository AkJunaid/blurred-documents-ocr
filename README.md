# Document Understanding

DeepSeek OCR to Markdown, structured extraction for retrieval, grounded drafting in one of five formats, and an operator feedback loop, with a FastAPI backend and Streamlit UI.

## Setup

### System dependencies (Linux)

The OCR pipeline relies on Poppler (for PDF rendering). Install it first:

```bash
sudo apt-get update
sudo apt-get install -y poppler-utils
```

### Python environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Python dependencies

Install everything from the lockstep requirements file:

```bash
pip install -r requirements.txt
```

Optional performance package (only if your CUDA stack supports it):

```bash
pip install flash-attn==2.7.3 --no-build-isolation
```

### Configure Groq

Set your Groq API key (or add it to a local environment file loaded by python-dotenv):

```bash
export GROQ_API_KEY="your-key-here"
```

## Docker (optional)

Docker build and compose files are in [Dockerfile](Dockerfile) and [docker-compose.yml](docker-compose.yml).

### Download via Docker (no git)

If you do not want to install git locally, you can use a lightweight container to clone the repo:

```bash
docker run --rm -v "$PWD":/work -w /work alpine/git clone <repo-url> document-understanding
cd document-understanding
```

### Build and run

```bash
export GROQ_API_KEY="your-key-here"
docker compose up --build
```

- API: http://localhost:8000
- UI: http://localhost:8501

The compose setup mounts [data](data) into the container to persist inputs and outputs.

### GPU support (required for DeepSeek OCR)

The OCR model in [deepseek.py](deepseek.py) runs on CUDA. With an NVIDIA GPU and the Nvidia Container Toolkit installed, run:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

## Run

### FastAPI server

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Streamlit UI

```bash
streamlit run streamlit_app.py
```

Open the app, keep the API URL as `http://localhost:8000`, upload a PDF or image, and run the pipeline.

### CLI pipeline

OCR + retrieval-grounded draft (single file or directory):

```bash
python main.py --input <file-or-dir> --draft-type case_fact_summary
```

You can also point to a directory to process all PDFs (and bundle all images into a single PDF):

```bash
python main.py --input <input-dir> --draft-type notice_related_summary
```

If you omit `--input`, the default input directory is [data/input](data/input).

When processing a directory, all supported image files are bundled into a single PDF and saved alongside the OCR outputs.

Draft types accepted by `--draft-type`:

- `title_review_summary`
- `case_fact_summary`
- `notice_related_summary`
- `document_checklist`
- `first_pass_internal_memo`

### API endpoints

- `POST /process` - upload a document or select an input file, returns OCR markdown, draft markdown, and evidence
- `GET /inputs` - list available input files in [data/input](data/input)
- `POST /feedback` - submit operator edits to extract reusable rules
- `POST /generate-pdf` - render markdown draft to PDF (optionally save edited markdown)
- `GET /health` - health check

### Outputs

- OCR markdown and structured JSON land in [data/output](data/output) by default.
- Drafts are written next to the OCR markdown with .draft.md and .draft.json suffixes.
- Retrieval vectors persist locally in [data/chroma_db](data/chroma_db).

## Architecture overview (brief)

- Input PDF/image -> DeepSeek OCR -> OCR markdown -> chunked JSON -> retrieval -> grounded draft -> feedback loop -> PDF export.
- OCR is handled in [deepseek.py](deepseek.py) and orchestrated in [pipeline/ocr.py](pipeline/ocr.py).
- Retrieval is handled by [pipeline/retrieval.py](pipeline/retrieval.py) with a local ChromaDB store.
- Drafting is grounded in evidence by [pipeline/drafting.py](pipeline/drafting.py) using Groq.
- Operator edits are mined into reusable rules in [pipeline/feedback.py](pipeline/feedback.py).
- The FastAPI backend is in [api.py](api.py), and the Streamlit UI is in [streamlit_app.py](streamlit_app.py).

## Assumptions and tradeoffs

- Requires an external Groq API key; draft quality and uptime depend on network and model availability.
- DeepSeek OCR is GPU-heavy; CPU-only runs are slower and can bottleneck large PDFs.
- Retrieval quality depends on OCR fidelity and chunking; noisy scans reduce evidence recall.
- Local ChromaDB persistence is convenient but not multi-tenant or access-controlled.
- No authentication, rate limiting, or user management is built in.


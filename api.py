"""
FastAPI backend for DeepSeek OCR, drafting, and PDF generation.
"""

from __future__ import annotations

import os
import re
import shutil
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from pipeline.drafting import (
    DRAFT_MODEL,
    DRAFT_TYPES,
    build_retrieval_query,
    generate_draft,
)
from pipeline.feedback import capture_edit, load_rules
from pipeline.ocr import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    list_input_files,
    run_ocr_pipeline,
)
from pipeline.pdf_render import render_pdf
from pipeline.retrieval import DocumentRetriever

app = FastAPI(title="Document Drafting API")


class PdfRequest(BaseModel):
    draft: str
    title: Optional[str] = None
    output_md_path: Optional[str] = None


class FeedbackRequest(BaseModel):
    original: str
    edited: str
    draft_type: str


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.get("/inputs")
def list_inputs() -> dict:
    return {"files": list_input_files(DEFAULT_INPUT_DIR)}


def _sanitize_slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-")
    return cleaned or "draft"


def _build_draft_path(ocr_md_path: str, draft_type: str) -> str:
    base = os.path.splitext(ocr_md_path)[0]
    slug = _sanitize_slug(draft_type)
    return f"{base}.{slug}.draft.md"


@app.post("/process")
async def process_document(
    file: UploadFile = File(None),
    input_name: Optional[str] = Form(None),
    draft_type: str = Form(...),
    top_k: int = Form(8),
    dpi: int = Form(200),
    base_size: int = Form(1024),
    image_size: int = Form(640),
    model: str = Form(DRAFT_MODEL),
) -> dict:
    os.makedirs(DEFAULT_INPUT_DIR, exist_ok=True)
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

    if draft_type not in DRAFT_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported draft_type")

    input_path = None
    if file is not None:
        ext = os.path.splitext(file.filename or "")[1].lower()
        if not ext:
            raise HTTPException(status_code=400, detail="File must have an extension")

        file_id = uuid.uuid4().hex
        saved_name = f"{file_id}{ext}"
        input_path = os.path.join(DEFAULT_INPUT_DIR, saved_name)

        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    elif input_name:
        input_path = os.path.join(DEFAULT_INPUT_DIR, input_name)

    if input_path and not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="Input file not found")

    if not input_path:
        raise HTTPException(status_code=400, detail="Provide a file or input_name")

    ocr_results = run_ocr_pipeline(
        input_path=input_path,
        input_dir=DEFAULT_INPUT_DIR,
        output_dir=DEFAULT_OUTPUT_DIR,
        dpi=dpi,
        base_size=base_size,
        image_size=image_size,
    )

    if not ocr_results:
        raise HTTPException(status_code=500, detail="No OCR output generated")
    if len(ocr_results) != 1:
        raise HTTPException(
            status_code=400,
            detail="Multiple OCR outputs generated. Select a single input file.",
        )

    ocr_md_path = ocr_results[0]["md_path"]
    ocr_json_path = ocr_results[0]["json_path"]
    with open(ocr_md_path, encoding="utf-8") as f:
        ocr_markdown = f.read()

    retriever = DocumentRetriever()
    retriever.index(ocr_json_path)
    source_file = os.path.basename(ocr_md_path)
    query = build_retrieval_query(draft_type)
    passages = retriever.retrieve(query, n_results=top_k, source_file=source_file)

    rules = load_rules(draft_type)
    draft_result = generate_draft(
        draft_type=draft_type,
        passages=passages,
        feedback_rules=rules,
        model=model,
    )
    draft_markdown = draft_result.get("draft", "")
    draft_md_path = _build_draft_path(ocr_md_path, draft_type)
    with open(draft_md_path, "w", encoding="utf-8") as f:
        f.write(draft_markdown)

    return {
        "input_file": os.path.basename(input_path),
        "ocr_md_path": ocr_md_path,
        "ocr_json_path": ocr_json_path,
        "ocr_markdown": ocr_markdown,
        "draft_md_path": draft_md_path,
        "draft_markdown": draft_markdown,
        "evidence_passages": draft_result.get("all_evidence", []),
        "evidence_used": draft_result.get("evidence_used", []),
        "draft_type": draft_type,
        "model": model,
    }


@app.post("/feedback")
def submit_feedback(payload: FeedbackRequest) -> dict:
    if payload.draft_type not in DRAFT_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported draft_type")
    return capture_edit(payload.original, payload.edited, payload.draft_type)


@app.post("/generate-pdf")
def generate_pdf(payload: PdfRequest) -> Response:
    if payload.output_md_path:
        output_dir = os.path.dirname(payload.output_md_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(payload.output_md_path, "w", encoding="utf-8") as f:
            f.write(payload.draft)

    pdf_bytes = render_pdf(payload.draft, title=payload.title)
    headers = {"Content-Disposition": "attachment; filename=draft.pdf"}
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)

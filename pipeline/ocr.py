"""
DeepSeek OCR pipeline wrapper.
Converts PDFs or images into markdown outputs using deepseek.py,
then builds structured JSON for retrieval and drafting.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Optional

from deepseek import process_path, SUPPORTED_IMAGE_EXTS, SUPPORTED_PDF_EXTS

DEFAULT_INPUT_DIR = "/home/junaid/Document-Understanding/data/input"
DEFAULT_OUTPUT_DIR = "/home/junaid/Document-Understanding/data/output"


def list_input_files(input_dir: str = DEFAULT_INPUT_DIR) -> list[str]:
    if not os.path.isdir(input_dir):
        return []

    files = []
    for name in os.listdir(input_dir):
        ext = os.path.splitext(name)[1].lower()
        if ext in SUPPORTED_PDF_EXTS or ext in SUPPORTED_IMAGE_EXTS:
            files.append(name)

    return sorted(files)


def resolve_input_path(input_path: Optional[str], input_dir: str) -> str:
    if not input_path:
        return input_dir
    if os.path.isabs(input_path):
        return input_path
    return os.path.join(input_dir, input_path)


def _parse_page_number(line: str, default_page: int) -> int:
    parts = line.replace("#", "").strip().split()
    if len(parts) >= 2 and parts[0].lower() == "page":
        try:
            return int(parts[1])
        except ValueError:
            return default_page
    return default_page


def _finalize_page(page: dict) -> dict:
    text = "\n".join(page["lines"]).strip()
    unclear = "[no text extracted" in text.lower() or "[error processing" in text.lower()
    return {
        "page": page["page"],
        "title": page.get("title"),
        "text": text,
        "unclear": unclear,
    }


def _split_pages(markdown_text: str) -> list[dict]:
    pages: list[dict] = []
    current = {"page": 1, "title": None, "lines": []}

    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# Page "):
            if current["lines"]:
                pages.append(_finalize_page(current))
            page_num = _parse_page_number(stripped, len(pages) + 1)
            current = {
                "page": page_num,
                "title": stripped.lstrip("#").strip(),
                "lines": [],
            }
            continue

        if stripped.startswith("# Image:"):
            if current["lines"]:
                pages.append(_finalize_page(current))
            current = {
                "page": len(pages) + 1,
                "title": stripped.lstrip("#").strip(),
                "lines": [],
            }
            continue

        if stripped == "---":
            continue

        current["lines"].append(line)

    if current["lines"] or not pages:
        pages.append(_finalize_page(current))

    return pages


def _chunk_paragraphs(text: str) -> list[str]:
    paragraphs = []
    buffer: list[str] = []
    for line in text.splitlines():
        if line.strip():
            buffer.append(line.strip())
        else:
            if buffer:
                paragraphs.append(" ".join(buffer).strip())
                buffer = []
    if buffer:
        paragraphs.append(" ".join(buffer).strip())
    return [para for para in paragraphs if para]


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        snippet = text[start : start + chunk_size].strip()
        if snippet:
            chunks.append(snippet)
        start += max(1, chunk_size - overlap)
    return chunks


def build_structured_output(
    md_path: str,
    output_dir: Optional[str] = None,
    chunk_size: int = 800,
    overlap: int = 100,
) -> str:
    with open(md_path, encoding="utf-8") as f:
        markdown_text = f.read()

    pages = _split_pages(markdown_text)
    chunks = []
    for page in pages:
        if page.get("unclear") or not page.get("text"):
            continue
        section = page.get("title") or f"Page {page['page']}"
        for paragraph in _chunk_paragraphs(page.get("text", "")):
            for snippet in _chunk_text(paragraph, chunk_size, overlap):
                chunks.append(
                    {
                        "id": str(uuid.uuid4()),
                        "text": snippet,
                        "page": page.get("page"),
                        "section": section,
                        "chunk_type": "paragraph",
                    }
                )

    stats = {
        "pages": len(pages),
        "chunks": len(chunks),
        "unclear_pages": sum(1 for page in pages if page.get("unclear")),
        "empty_pages": sum(1 for page in pages if not page.get("text")),
    }

    payload = {
        "source_md": md_path,
        "text": markdown_text,
        "pages": pages,
        "chunks": chunks,
        "stats": stats,
    }

    output_dir = output_dir or str(Path(md_path).parent)
    json_path = os.path.join(
        output_dir,
        f"{Path(md_path).stem}.json",
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return json_path


def run_ocr_pipeline(
    input_path: Optional[str] = None,
    input_dir: str = DEFAULT_INPUT_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    dpi: int = 200,
    base_size: int = 1024,
    image_size: int = 640,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[dict]:
    resolved_input = resolve_input_path(input_path, input_dir)
    md_paths = process_path(
        resolved_input,
        output_dir,
        dpi=dpi,
        base_size=base_size,
        image_size=image_size,
    )

    results = []
    for md_path in md_paths:
        json_path = build_structured_output(
            md_path,
            output_dir=output_dir,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        results.append({"md_path": md_path, "json_path": json_path})

    return results

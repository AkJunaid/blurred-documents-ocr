"""
CLI: DeepSeek OCR -> draft markdown.
"""

from __future__ import annotations

import argparse
import json
import os

from pipeline.drafting import (
    DRAFT_MODEL,
    DRAFT_TYPES,
    build_retrieval_query,
    generate_draft,
)
from pipeline.feedback import load_rules
from pipeline.ocr import DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, run_ocr_pipeline
from pipeline.retrieval import DocumentRetriever


def _sanitize_slug(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value.lower())
    cleaned = "-".join(segment for segment in cleaned.split("-") if segment)
    return cleaned or "draft"


def _build_draft_path(ocr_md_path: str, draft_type: str) -> str:
    base = os.path.splitext(ocr_md_path)[0]
    return f"{base}.{_sanitize_slug(draft_type)}.draft.md"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek OCR -> Draft pipeline")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_DIR,
        help="Input file or directory (default: data/input)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for OCR and draft markdown",
    )
    parser.add_argument(
        "--draft-type",
        required=True,
        choices=list(DRAFT_TYPES.keys()),
        help="Draft type to generate",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--base-size", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--model", default=DRAFT_MODEL)
    parser.add_argument(
        "--output-draft",
        default=None,
        help="Output draft markdown path (only valid for single input)",
    )
    parser.add_argument(
        "--output-draft-json",
        default=None,
        help="Output draft JSON path (only valid for single input)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    ocr_results = run_ocr_pipeline(
        input_path=args.input,
        input_dir=DEFAULT_INPUT_DIR,
        output_dir=args.output_dir,
        dpi=args.dpi,
        base_size=args.base_size,
        image_size=args.image_size,
    )

    if not ocr_results:
        raise SystemExit("No OCR markdown generated")

    if args.output_draft and len(ocr_results) != 1:
        raise SystemExit("--output-draft requires a single input file")
    if args.output_draft_json and len(ocr_results) != 1:
        raise SystemExit("--output-draft-json requires a single input file")

    retriever = DocumentRetriever()
    query = build_retrieval_query(args.draft_type)

    for result in ocr_results:
        ocr_md_path = result["md_path"]
        ocr_json_path = result["json_path"]

        retriever.index(ocr_json_path)
        source_file = os.path.basename(ocr_md_path)
        passages = retriever.retrieve(query, n_results=args.top_k, source_file=source_file)
        rules = load_rules(args.draft_type)

        draft_result = generate_draft(
            draft_type=args.draft_type,
            passages=passages,
            feedback_rules=rules,
            model=args.model,
        )
        draft_text = draft_result.get("draft", "")

        draft_path = args.output_draft or _build_draft_path(
            ocr_md_path, args.draft_type
        )
        output_dir = os.path.dirname(draft_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(draft_path, "w", encoding="utf-8") as f:
            f.write(draft_text)

        draft_json_path = args.output_draft_json or draft_path.replace(
            ".draft.md", ".draft.json"
        )
        with open(draft_json_path, "w", encoding="utf-8") as f:
            json.dump(draft_result, f, indent=2)

        print(f"Draft written to {draft_path}")
        print(f"Draft evidence written to {draft_json_path}")


if __name__ == "__main__":
    main()

"""
Streamlit UI for DeepSeek OCR, draft generation, editing, and PDF export.
"""

from __future__ import annotations

import requests
import streamlit as st

st.set_page_config(page_title="Document Drafting", layout="wide")

st.title("Document Drafting")

api_url = st.sidebar.text_input("API URL", "http://localhost:8000")

input_mode = st.sidebar.radio(
    "Input mode",
    ["Upload file", "Use existing input file"],
)

uploaded = None
selected_file = None
if input_mode == "Upload file":
    uploaded = st.file_uploader(
        "Upload a PDF or image",
        type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"],
    )
else:
    try:
        response = requests.get(f"{api_url}/inputs", timeout=10)
        if response.ok:
            files = response.json().get("files", [])
        else:
            files = []
            st.sidebar.error("Failed to fetch input files from API")
    except Exception:
        files = []
        st.sidebar.error("API not reachable to list input files")

    selected_file = st.selectbox("Select an input file", options=files)

draft_type_labels = {
    "Title Review Summary": "title_review_summary",
    "Case Fact Summary": "case_fact_summary",
    "Notice-Related Summary": "notice_related_summary",
    "Document Checklist": "document_checklist",
    "First-Pass Internal Memo": "first_pass_internal_memo",
}

draft_type_label = st.selectbox("Draft type", list(draft_type_labels.keys()))
draft_type = draft_type_labels[draft_type_label]

col1, col2, col3 = st.columns(3)
with col1:
    dpi = st.number_input("DPI", min_value=150, max_value=600, value=200, step=50)
with col2:
    base_size = st.number_input("Base size", min_value=512, max_value=2048, value=1024, step=128)
with col3:
    image_size = st.number_input("Image size", min_value=256, max_value=1024, value=640, step=64)

top_k = st.number_input("Evidence passages (top-k)", min_value=3, max_value=30, value=8, step=1)

model = st.text_input("Groq model", "llama-3.3-70b-versatile")

if st.button("Run OCR + Draft", type="primary"):
    if input_mode == "Upload file" and not uploaded:
        st.warning("Please upload a PDF or image first.")
    elif input_mode == "Use existing input file" and not selected_file:
        st.warning("Please select an input file.")
    else:
        with st.spinner("Running DeepSeek OCR and draft generation..."):
            files = None
            data = {
                "draft_type": draft_type,
                "top_k": str(top_k),
                "dpi": str(dpi),
                "base_size": str(base_size),
                "image_size": str(image_size),
                "model": model,
            }
            if input_mode == "Upload file":
                files = {
                    "file": (uploaded.name, uploaded.getvalue(), uploaded.type),
                }
            else:
                data["input_name"] = selected_file

            response = requests.post(f"{api_url}/process", files=files, data=data)

        if response.ok:
            payload = response.json()
            st.session_state["ocr_markdown"] = payload.get("ocr_markdown", "")
            st.session_state["draft_markdown"] = payload.get("draft_markdown", "")
            st.session_state["ocr_md_path"] = payload.get("ocr_md_path")
            st.session_state["ocr_json_path"] = payload.get("ocr_json_path")
            st.session_state["draft_md_path"] = payload.get("draft_md_path")
            st.session_state["draft_type_label"] = draft_type_label
            st.session_state["draft_type"] = draft_type
            st.session_state["evidence_passages"] = payload.get("evidence_passages", [])
            st.session_state["evidence_used"] = payload.get("evidence_used", [])
            st.success("Done")
        else:
            st.error(f"API error: {response.status_code} - {response.text}")

if "draft_markdown" in st.session_state:
    original_draft = st.session_state.get("draft_markdown", "")

    st.subheader("Draft")
    edited_draft = st.text_area(
        "Edit the draft before exporting",
        value=st.session_state.get("edited_draft", original_draft),
        height=350,
    )
    st.session_state["edited_draft"] = edited_draft

    st.download_button(
        "Download OCR Markdown",
        data=st.session_state.get("ocr_markdown", ""),
        file_name="ocr_output.md",
        mime="text/markdown",
    )
    st.download_button(
        "Download Draft Markdown",
        data=edited_draft,
        file_name="draft_output.md",
        mime="text/markdown",
    )

    if st.session_state.get("ocr_markdown"):
        with st.expander("OCR Markdown"):
            st.text_area(
                "OCR output",
                value=st.session_state.get("ocr_markdown", ""),
                height=250,
            )

    evidence_used = st.session_state.get("evidence_used", [])
    if evidence_used:
        with st.expander("Evidence used"):
            for item in evidence_used:
                page = item.get("page")
                section = item.get("section")
                score = item.get("relevance_score")
                st.write(f"Page {page} | Section: {section} | Score: {score}")
                st.write(item.get("text"))

    evidence_passages = st.session_state.get("evidence_passages", [])
    if evidence_passages:
        with st.expander("All retrieved evidence"):
            for item in evidence_passages:
                page = item.get("page")
                section = item.get("section")
                score = item.get("relevance_score")
                st.write(f"Page {page} | Section: {section} | Score: {score}")
                st.write(item.get("text"))

    if st.button("Submit edits and learn"):
        payload = {
            "original": original_draft,
            "edited": edited_draft,
            "draft_type": st.session_state.get("draft_type", draft_type),
        }
        response = requests.post(f"{api_url}/feedback", json=payload)
        if response.ok:
            data = response.json()
            st.success(f"Rules extracted: {data.get('rules_extracted', [])}")
        else:
            st.error(f"Feedback error: {response.status_code} - {response.text}")

    if st.button("Generate PDF"):
        draft_md_path = st.session_state.get("draft_md_path") or "draft_output.md"
        if draft_md_path.endswith(".draft.md"):
            final_md_path = draft_md_path.replace(".draft.md", ".final.md")
        else:
            final_md_path = draft_md_path.replace(".md", ".final.md")

        payload = {
            "draft": edited_draft,
            "title": st.session_state.get("draft_type_label", "Draft Output"),
            "output_md_path": final_md_path,
        }
        response = requests.post(f"{api_url}/generate-pdf", json=payload)
        if response.ok:
            st.session_state["pdf_bytes"] = response.content
            st.success(f"Saved edited draft to {final_md_path}")
        else:
            st.error(f"PDF error: {response.status_code} - {response.text}")

    if "pdf_bytes" in st.session_state:
        st.download_button(
            "Download Draft PDF",
            data=st.session_state["pdf_bytes"],
            file_name="draft_output.pdf",
            mime="application/pdf",
        )

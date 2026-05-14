"""
Grounded draft generation using retrieved evidence.
Every claim in the draft is anchored to source passages.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Optional

from groq import Groq

try:
    from dotenv import load_dotenv
except ImportError:  # Optional dependency.
    load_dotenv = None

if load_dotenv:
    load_dotenv()

DRAFT_MODEL = "llama-3.3-70b-versatile"

DRAFT_TYPES = {
    "title_review_summary": {
        "label": "Title Review Summary",
        "instructions": (
            "Summarize title and ownership-related information in the document. "
            "Highlight parties, property/asset identifiers, title status, encumbrances, "
            "exceptions, and any risks or missing items."
        ),
        "retrieval_query": "title ownership deed lien encumbrance exception property asset parties",
    },
    "case_fact_summary": {
        "label": "Case Fact Summary",
        "instructions": (
            "Summarize the key facts, parties, dates, and events in a neutral tone. "
            "Avoid legal conclusions and stick to facts present in the OCR text."
        ),
        "retrieval_query": "facts parties dates events chronology key terms",
    },
    "notice_related_summary": {
        "label": "Notice-Related Summary",
        "instructions": (
            "Focus on notice provisions, deadlines, service methods, recipients, "
            "addresses, and any time-sensitive actions."
        ),
        "retrieval_query": "notice deadline service method recipients address time-sensitive",
    },
    "document_checklist": {
        "label": "Document Checklist",
        "instructions": (
            "Create a checklist of key items, fields, and supporting documents "
            "that should be confirmed, with missing items clearly called out."
        ),
        "retrieval_query": "checklist required fields missing items documents exhibits attachments",
    },
    "first_pass_internal_memo": {
        "label": "First-Pass Internal Memo",
        "instructions": (
            "Write an internal memo with sections for overview, key terms, risks, "
            "open questions, and recommended next steps."
        ),
        "retrieval_query": "overview key terms risks questions next steps",
    },
}


@lru_cache(maxsize=1)
def _get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")
    return Groq(api_key=api_key)


def get_draft_type_options() -> list[str]:
    return list(DRAFT_TYPES.keys())


def build_retrieval_query(draft_type: str) -> str:
    if draft_type not in DRAFT_TYPES:
        raise ValueError(f"Unsupported draft_type: {draft_type}")
    query = DRAFT_TYPES[draft_type].get("retrieval_query")
    if query:
        return query
    return f"{DRAFT_TYPES[draft_type]['label']}. {DRAFT_TYPES[draft_type]['instructions']}"


def _format_evidence(passages: list[dict]) -> str:
    """Format retrieved passages as a numbered evidence block for the LLM."""
    if not passages:
        return "[none]"

    lines = []
    for i, passage in enumerate(passages, 1):
        lines.append(
            f"[{i}] (page={passage.get('page')}, section={passage.get('section')}, "
            f"score={passage.get('relevance_score')})\n"
            f"    {passage.get('text')}"
        )
    return "\n\n".join(lines)


def generate_draft(
    draft_type: str,
    passages: list[dict],
    feedback_rules: Optional[list[str]] = None,
    model: str = DRAFT_MODEL,
) -> dict:
    """
    Generate a grounded draft for a given draft type.

    Returns:
        {
            "draft": str,
            "evidence_used": list[dict],
            "all_evidence": list[dict],
        }
    """
    if draft_type not in DRAFT_TYPES:
        raise ValueError(f"Unsupported draft_type: {draft_type}")

    if not passages:
        return {
            "draft": "No evidence retrieved. Refine the query or index more documents.",
            "evidence_used": [],
            "all_evidence": [],
            "draft_type": draft_type,
        }

    draft_label = DRAFT_TYPES[draft_type]["label"]
    instructions = DRAFT_TYPES[draft_type]["instructions"]
    evidence_block = _format_evidence(passages)

    rules_block = ""
    if feedback_rules:
        cleaned = [rule.strip() for rule in feedback_rules if rule and rule.strip()]
        if cleaned:
            rules_block = (
                "\n\nOperator style rules learned from past edits:\n"
                + "\n".join(f"- {rule}" for rule in cleaned)
            )

    system_prompt = (
        "You are a legal drafting assistant.\n"
        "You generate draft responses grounded ONLY in the provided evidence passages.\n\n"
        "Rules:\n"
        "- Use ONLY the evidence provided. Do not invent facts.\n"
        "- After every claim, add an inline citation like [1] or [2].\n"
        "- If evidence is insufficient for part of the draft, say so explicitly.\n"
        "- Structure the draft clearly with sections where appropriate.\n"
        "- Do not add legal boilerplate not supported by the evidence."
    ).strip() + rules_block

    user_message = (
        f"Draft type: {draft_label}\n"
        f"Instructions: {instructions}\n\n"
        "Evidence passages (retrieved from source documents):\n"
        f"{evidence_block}\n\n"
        "Write a grounded draft in Markdown. Cite evidence inline using [1], [2], etc."
    )

    response = _get_groq_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        max_tokens=2000,
    )

    draft_text = response.choices[0].message.content.strip()
    cited_nums = {int(n) for n in re.findall(r"\[(\d+)\]", draft_text)}
    evidence_used = [
        passage
        for i, passage in enumerate(passages, 1)
        if i in cited_nums
    ]

    return {
        "draft": draft_text,
        "evidence_used": evidence_used,
        "all_evidence": passages,
        "draft_type": draft_type,
        "model": model,
    }

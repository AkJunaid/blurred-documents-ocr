"""
Operator feedback loop.
Captures edits, extracts reusable rules, and improves future drafts.
"""

from __future__ import annotations

import difflib
import json
import os
import re
from datetime import datetime
from functools import lru_cache
from typing import Optional

from groq import Groq

try:
    from dotenv import load_dotenv
except ImportError:  # Optional dependency.
    load_dotenv = None

if load_dotenv:
    load_dotenv()

FEEDBACK_PATH = "./data/feedback/rules.json"
MODEL = "llama-3.3-70b-versatile"


@lru_cache(maxsize=1)
def _get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")
    return Groq(api_key=api_key)


def capture_edit(original_draft: str, edited_draft: str, draft_type: str) -> dict:
    """
    Diff original vs edited draft, then extract reusable rules with Groq.
    Rules are saved to disk for future drafts.
    """
    diff_lines = list(
        difflib.unified_diff(
            original_draft.splitlines(),
            edited_draft.splitlines(),
            lineterm="",
            n=2,
        )
    )
    diff_text = "\n".join(diff_lines)

    if not diff_text.strip():
        return {"rules_extracted": [], "message": "No changes detected"}

    response = _get_groq_client().chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract reusable drafting rules from operator edits.\n"
                    "Given a diff of an original vs edited draft and the draft type, "
                    "identify what the operator changed and why.\n"
                    "Return ONLY a JSON array of short, actionable rules (strings).\n"
                    "Rules should be general enough to apply to future drafts of the same type.\n"
                    "No explanation, no markdown. Just the JSON array."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Draft type: {draft_type}\n\n"
                    "Diff (lines starting with - were removed, + were added):\n"
                    f"{diff_text}\n\n"
                    "Extract reusable drafting rules from these edits."
                ),
            },
        ],
        temperature=0.1,
        max_tokens=500,
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        new_rules = json.loads(raw)
    except json.JSONDecodeError:
        new_rules = []

    if not isinstance(new_rules, list):
        new_rules = []
    new_rules = [rule for rule in new_rules if isinstance(rule, str) and rule.strip()]

    _save_rules(new_rules, draft_type)

    return {
        "rules_extracted": new_rules,
        "diff_summary": f"{len(diff_lines)} lines changed",
    }


def _save_rules(new_rules: list[str], draft_type: str) -> None:
    os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)

    existing: list[dict] = []
    if os.path.exists(FEEDBACK_PATH):
        with open(FEEDBACK_PATH, encoding="utf-8") as f:
            existing = json.load(f)
        if not isinstance(existing, list):
            existing = []

    for rule in new_rules:
        existing.append(
            {
                "rule": rule,
                "draft_type": draft_type,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)


def load_rules(draft_type: Optional[str] = None) -> list[str]:
    """Load accumulated operator rules for a draft type (or all)."""
    if not os.path.exists(FEEDBACK_PATH):
        return []
    with open(FEEDBACK_PATH, encoding="utf-8") as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        return []

    rules = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if draft_type and entry.get("draft_type") != draft_type:
            continue
        rule = entry.get("rule")
        if rule:
            rules.append(rule)

    return rules

"""
PDF rendering for draft text.
"""

from __future__ import annotations

from io import BytesIO
from typing import Optional

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer


def render_pdf(draft_text: str, title: Optional[str] = None) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    style_body = styles["BodyText"]
    style_h1 = styles["Heading1"]
    style_h2 = styles["Heading2"]
    style_bullet = ParagraphStyle(
        "Bullet",
        parent=style_body,
        leftIndent=18,
        bulletIndent=9,
    )

    story = []
    if title:
        story.append(Paragraph(title, style_h1))
        story.append(Spacer(1, 0.2 * inch))

    bullet_items: list[ListItem] = []

    def flush_bullets() -> None:
        if bullet_items:
            story.append(ListFlowable(bullet_items, bulletType="bullet"))
            bullet_items.clear()

    for raw_line in draft_text.splitlines():
        line = raw_line.strip()
        if not line:
            flush_bullets()
            story.append(Spacer(1, 0.15 * inch))
            continue

        if line.startswith("# "):
            flush_bullets()
            story.append(Paragraph(line[2:].strip(), style_h1))
            continue
        if line.startswith("## "):
            flush_bullets()
            story.append(Paragraph(line[3:].strip(), style_h2))
            continue
        if line.startswith("- "):
            bullet_items.append(ListItem(Paragraph(line[2:].strip(), style_body)))
            continue

        flush_bullets()
        story.append(Paragraph(line, style_body))

    flush_bullets()
    doc.build(story)
    return buffer.getvalue()

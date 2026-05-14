from __future__ import annotations

from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "report.md"
TARGET = ROOT / "docs" / "Financial_ML_Track_Report.pdf"
EDA_IMAGES = [
    ROOT / "outputs" / "eda_missingness.png",
    ROOT / "outputs" / "eda_correlation_heatmap.png",
]


def clean_inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    text = text.replace("**", "")
    return text


def parse_markdown_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    rows = []
    index = start
    while index < len(lines) and lines[index].strip().startswith("|"):
        line = lines[index].strip()
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not all(set(cell) <= {":", "-"} for cell in cells):
            rows.append(cells)
        index += 1
    return rows, index


def build_pdf() -> None:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontSize=8.2,
            leading=10.1,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Tiny",
            parent=styles["BodyText"],
            fontSize=6.5,
            leading=7.4,
        )
    )
    styles["Title"].fontSize = 16
    styles["Title"].leading = 18
    styles["Heading2"].fontSize = 11
    styles["Heading2"].leading = 13
    styles["Heading2"].spaceBefore = 5
    styles["Heading2"].spaceAfter = 3

    doc = SimpleDocTemplate(
        str(TARGET),
        pagesize=A4,
        rightMargin=0.45 * inch,
        leftMargin=0.45 * inch,
        topMargin=0.42 * inch,
        bottomMargin=0.42 * inch,
    )

    story = []
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()
        if not stripped:
            index += 1
            continue

        if stripped.startswith("# "):
            story.append(Paragraph(clean_inline(stripped[2:]), styles["Title"]))
            story.append(Spacer(1, 4))
        elif stripped.startswith("## "):
            heading = stripped[3:]
            if heading.startswith("6. "):
                story.append(PageBreak())
            story.append(Paragraph(clean_inline(heading), styles["Heading2"]))
        elif stripped.startswith("|"):
            rows, index = parse_markdown_table(lines, index)
            if rows:
                table_data = [
                    [Paragraph(clean_inline(cell), styles["Tiny"]) for cell in row]
                    for row in rows
                ]
                table = Table(table_data, repeatRows=1, hAlign="LEFT")
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 2),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                            ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
                        ]
                    )
                )
                story.append(table)
                story.append(Spacer(1, 3))
            continue
        elif stripped.startswith("- "):
            story.append(Paragraph("• " + clean_inline(stripped[2:]), styles["Small"]))
        else:
            story.append(Paragraph(clean_inline(stripped), styles["Small"]))
            if stripped.startswith("Scaling decision:"):
                story.append(Spacer(1, 4))
                for image_path in EDA_IMAGES:
                    if image_path.exists():
                        story.append(Image(str(image_path), width=5.8 * inch, height=3.2 * inch))
                        story.append(Spacer(1, 4))
        index += 1

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story)


if __name__ == "__main__":
    build_pdf()
    print(TARGET)

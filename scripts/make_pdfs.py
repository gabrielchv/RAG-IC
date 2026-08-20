"""Render the markdown sources in ``data/<company>/pdf_sources/`` to PDFs.

Why bother: a corpus of only markdown would let the ingestion pipeline off the hook.
Real internal documents arrive as PDFs, and PDF text extraction introduces noise —
broken line wraps, lost heading structure, page-boundary splits — that shows up in
the retrieval metrics. Generating the PDFs from the same source text means that noise
can be measured rather than guessed at: the content is identical, only the container
differs.

Usage::

    python scripts/make_pdfs.py --company aurora

Requires the dev extra: ``pip install -e ".[dev]"``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
#: reportlab's Paragraph parses a mini-HTML dialect, so bare & < > must be escaped.
_ESCAPES = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}


def _escape(text: str) -> str:
    for char, entity in _ESCAPES.items():
        text = text.replace(char, entity)
    return text


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "DocTitle", parent=base["Title"], fontSize=15, spaceAfter=14, alignment=0
        ),
        "h2": ParagraphStyle("DocH2", parent=base["Heading2"], fontSize=11.5, spaceBefore=12),
        "h3": ParagraphStyle("DocH3", parent=base["Heading3"], fontSize=10.5, spaceBefore=8),
        "body": ParagraphStyle(
            "DocBody", parent=base["BodyText"], fontSize=9.5, leading=14, alignment=TA_JUSTIFY
        ),
    }


def markdown_to_flowables(markdown: str, styles: dict[str, ParagraphStyle]) -> list:
    flowables: list = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            flowables.append(Paragraph(_escape(" ".join(paragraph)), styles["body"]))
            flowables.append(Spacer(1, 5))
            paragraph.clear()

    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        heading = _HEADING_RE.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            style = "title" if level == 1 else "h2" if level == 2 else "h3"
            flowables.append(Paragraph(_escape(heading.group(2)), styles[style]))
            continue
        if not line:
            flush()
            continue
        paragraph.append(line)

    flush()
    return flowables


def render(source: Path, destination: Path, styles: dict[str, ParagraphStyle]) -> None:
    title = source.stem
    first_line = source.read_text(encoding="utf-8").splitlines()[0] if source.stat().st_size else ""
    heading = _HEADING_RE.match(first_line.strip())
    if heading and len(heading.group(1)) == 1:
        title = heading.group(2).strip()

    destination.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(destination),
        pagesize=A4,
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
        topMargin=2.0 * cm,
        bottomMargin=2.0 * cm,
        title=title,
        author="Metalúrgica Aurora Ltda.",
    )
    document.build(markdown_to_flowables(source.read_text(encoding="utf-8"), styles))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company", default="aurora")
    args = parser.parse_args()

    sources_dir = PROJECT_ROOT / "data" / args.company / "pdf_sources"
    output_dir = PROJECT_ROOT / "data" / args.company / "docs" / "pdf"

    sources = sorted(sources_dir.glob("*.md"))
    if not sources:
        raise SystemExit(f"no markdown sources in {sources_dir}")

    styles = _styles()
    for source in sources:
        destination = output_dir / f"{source.stem}.pdf"
        render(source, destination, styles)
        print(f"{source.name} -> {destination.relative_to(PROJECT_ROOT)}")

    print(f"\n{len(sources)} PDFs gerados em {output_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

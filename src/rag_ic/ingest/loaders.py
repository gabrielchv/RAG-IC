"""Per-format loaders turning source files into :class:`Document` objects.

Every loader's job is the same: produce a flat list of :class:`Block` objects that
know *where they came from*. Chunking happens later and is format-agnostic — which
is what keeps the chunker testable without any file on disk.
"""

from __future__ import annotations

import csv
import logging
import re
from collections.abc import Iterator
from pathlib import Path

from rag_ic.models import Block, Document, SourceFormat

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES: dict[str, SourceFormat] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "markdown",
    ".pdf": "pdf",
    ".csv": "csv",
}

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_TABLE_ROW_RE = re.compile(r"^\s*\|")


def doc_id_for(path: Path, root: Path) -> str:
    """Stable identifier: the path relative to the corpus root, without suffix."""
    return path.relative_to(root).with_suffix("").as_posix()


def _humanise(stem: str) -> str:
    return stem.replace("-", " ").replace("_", " ").strip().capitalize()


# --------------------------------------------------------------------------- markdown


def _markdown_blocks(lines: list[str]) -> Iterator[tuple[tuple[str, ...], str, bool]]:
    """Yield ``(section_path, text, atomic)`` for each paragraph-level unit.

    The document's ``# H1`` is treated as the title and excluded from section paths;
    ``##`` and deeper build the path.
    """
    heading_stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    buffer_atomic = False
    in_code_fence = False

    def current_path() -> tuple[str, ...]:
        return tuple(title for level, title in heading_stack if level >= 2)

    def flush() -> Iterator[tuple[tuple[str, ...], str, bool]]:
        nonlocal buffer, buffer_atomic
        if buffer:
            text = "\n".join(buffer).strip()
            if text:
                yield current_path(), text, buffer_atomic
        buffer = []
        buffer_atomic = False

    for raw in lines:
        line = raw.rstrip("\n")

        if line.strip().startswith("```"):
            in_code_fence = not in_code_fence
            buffer.append(line)
            buffer_atomic = True
            if not in_code_fence:
                yield from flush()
            continue

        if in_code_fence:
            buffer.append(line)
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            yield from flush()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            continue

        if not line.strip():
            yield from flush()
            continue

        # Keep a whole markdown table together: splitting one is worse than a long chunk.
        if _TABLE_ROW_RE.match(line):
            if buffer and not buffer_atomic:
                yield from flush()
            buffer.append(line)
            buffer_atomic = True
            continue
        if buffer_atomic and not _TABLE_ROW_RE.match(line):
            yield from flush()

        buffer.append(line)

    yield from flush()


def load_markdown(path: Path, doc_id: str) -> Document:
    lines = path.read_text(encoding="utf-8").splitlines()

    title = _humanise(path.stem)
    for line in lines:
        heading = _HEADING_RE.match(line)
        if heading and len(heading.group(1)) == 1:
            title = heading.group(2).strip()
            break

    blocks = [
        Block(text=text, section_path=section_path, atomic=atomic)
        for section_path, text, atomic in _markdown_blocks(lines)
    ]
    return Document(
        doc_id=doc_id,
        title=title,
        source_path=path.as_posix(),
        source_format="markdown",
        blocks=blocks,
    )


# --------------------------------------------------------------------------- pdf


def load_pdf(path: Path, doc_id: str) -> Document:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    title = _humanise(path.stem)
    if reader.metadata and reader.metadata.title:
        title = reader.metadata.title.strip() or title

    blocks: list[Block] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        # pypdf wraps hard at the layout's line width; rejoin lines that are
        # clearly a continued sentence so the sentence splitter can do its job.
        text = re.sub(r"(?<![.:;!?])\n(?=[a-zà-ÿ(])", " ", text)
        for paragraph in re.split(r"\n\s*\n", text):
            cleaned = paragraph.strip()
            if cleaned:
                blocks.append(Block(text=cleaned, page=page_number))

    if not blocks:
        logger.warning("PDF %s produced no extractable text", path)

    return Document(
        doc_id=doc_id,
        title=title,
        source_path=path.as_posix(),
        source_format="pdf",
        blocks=blocks,
    )


# --------------------------------------------------------------------------- csv


def load_csv(path: Path, doc_id: str) -> Document:
    """One block per row, rendered as ``column: value`` pairs.

    Embedding a whole table is useless — the row is the semantic unit here, and a
    row rendered as prose is something both the embedder and BM25 can match on.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        blocks = [
            Block(
                text="; ".join(
                    f"{column}: {(row.get(column) or '').strip()}"
                    for column in fieldnames
                    if (row.get(column) or "").strip()
                ),
                section_path=(f"linha {row_number}",),
                row=row_number,
                atomic=True,
            )
            for row_number, row in enumerate(reader, start=1)
        ]

    blocks = [block for block in blocks if block.text]
    return Document(
        doc_id=doc_id,
        title=_humanise(path.stem),
        source_path=path.as_posix(),
        source_format="csv",
        blocks=blocks,
    )


# --------------------------------------------------------------------------- dispatch


_LOADERS = {"markdown": load_markdown, "pdf": load_pdf, "csv": load_csv}


def load_document(path: Path, root: Path) -> Document:
    """Load one file, dispatching on its suffix."""
    source_format = SUPPORTED_SUFFIXES.get(path.suffix.lower())
    if source_format is None:
        raise ValueError(f"unsupported file type: {path}")
    return _LOADERS[source_format](path, doc_id_for(path, root))


def load_corpus(root: Path) -> list[Document]:
    """Load every supported file under ``root``, sorted for a deterministic index."""
    if not root.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {root}")

    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    documents = []
    for path in paths:
        document = load_document(path, root)
        logger.info(
            "loaded %-45s %-8s %3d blocos",
            document.doc_id,
            document.source_format,
            len(document.blocks),
        )
        documents.append(document)
    return documents

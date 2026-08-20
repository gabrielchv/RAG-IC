"""Structure-aware chunking with a sentence-level fallback.

Strategy, in order of preference:

1. Keep a document's own structure. Blocks under the same heading (same page, same
   row) merge together while they fit the token budget.
2. If one block alone exceeds the budget, split it on sentence boundaries, carrying
   an overlap of roughly ``overlap_ratio`` of the budget into the next piece so a
   fact straddling the cut survives in at least one chunk.
3. If a single sentence still exceeds the budget — a malformed table row, a PDF
   extraction artefact — hard-split it on whitespace. Rare, but it must not hang.

Every chunk is later indexed together with its ``context_header`` (document title +
heading path), so the budget reserves room for that header up front.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from rag_ic.ingest.tokens import TokenCounter, approximate_token_count
from rag_ic.models import Block, Chunk, Document

logger = logging.getLogger(__name__)

#: Sentence boundary: punctuation followed by whitespace, or a hard line break.
#: Kept deliberately simple — pt-BR abbreviations ("art.", "nº") are rare enough in
#: this corpus that a heavier splitter would not pay for itself.
_SENTENCE_RE = re.compile(r"(?<=[.!?;:])\s+|\n+")


@dataclass(frozen=True)
class ChunkingConfig:
    max_tokens: int = 400
    overlap_ratio: float = 0.15
    #: Tokens reserved for the context header prepended at index time.
    header_reserve_tokens: int = 40

    @property
    def overlap_tokens(self) -> int:
        return max(0, int(self.max_tokens * self.overlap_ratio))


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_RE.split(text) if part.strip()]


def _hard_split(text: str, budget: int, count_tokens: TokenCounter) -> list[str]:
    """Last resort: split on whitespace until each piece fits."""
    words = text.split()
    pieces: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = [*current, word]
        if current and count_tokens(" ".join(candidate)) > budget:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current = candidate
    if current:
        pieces.append(" ".join(current))
    return pieces or [text]


def split_long_text(text: str, budget: int, overlap: int, count_tokens: TokenCounter) -> list[str]:
    """Split ``text`` into pieces of at most ``budget`` tokens with sentence overlap."""
    if count_tokens(text) <= budget:
        return [text]

    sentences = _split_sentences(text)
    pieces: list[str] = []
    current: list[str] = []

    def current_tokens() -> int:
        return count_tokens(" ".join(current))

    for sentence in sentences:
        if count_tokens(sentence) > budget:
            if current:
                pieces.append(" ".join(current))
                current = []
            pieces.extend(_hard_split(sentence, budget, count_tokens))
            continue

        current.append(sentence)
        if current_tokens() > budget:
            current.pop()
            pieces.append(" ".join(current))
            current = [*_tail_within(current, overlap, count_tokens), sentence]

    if current:
        pieces.append(" ".join(current))
    return pieces


def _tail_within(sentences: list[str], overlap: int, count_tokens: TokenCounter) -> list[str]:
    """Trailing sentences whose combined length stays within ``overlap`` tokens."""
    if overlap <= 0:
        return []
    tail: list[str] = []
    for sentence in reversed(sentences):
        candidate = [sentence, *tail]
        if count_tokens(" ".join(candidate)) > overlap:
            break
        tail = candidate
    return tail


def _group_key(block: Block) -> tuple[tuple[str, ...], int | None]:
    return block.section_path, block.page


def chunk_document(
    document: Document,
    config: ChunkingConfig | None = None,
    count_tokens: TokenCounter | None = None,
) -> list[Chunk]:
    """Turn one loaded document into indexable chunks."""
    config = config or ChunkingConfig()
    count_tokens = count_tokens or approximate_token_count
    budget = max(32, config.max_tokens - config.header_reserve_tokens)

    chunks: list[Chunk] = []
    ordinal = 0

    def emit(text: str, block: Block) -> None:
        nonlocal ordinal
        cleaned = text.strip()
        if not cleaned:
            return
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}#{ordinal:04d}",
                doc_id=document.doc_id,
                doc_title=document.title,
                source_path=document.source_path,
                source_format=document.source_format,
                section_path=block.section_path,
                text=cleaned,
                token_count=count_tokens(cleaned),
                ordinal=ordinal,
                page=block.page,
                row=block.row,
            )
        )
        ordinal += 1

    pending: list[Block] = []

    def flush() -> None:
        if not pending:
            return
        merged = "\n\n".join(block.text for block in pending)
        head = pending[0]
        for piece in split_long_text(merged, budget, config.overlap_tokens, count_tokens):
            emit(piece, head)
        pending.clear()

    for block in document.blocks:
        if pending and (_group_key(pending[0]) != _group_key(block) or block.atomic):
            flush()

        if block.atomic:
            for piece in split_long_text(block.text, budget, config.overlap_tokens, count_tokens):
                emit(piece, block)
            continue

        candidate = "\n\n".join([*(b.text for b in pending), block.text])
        if pending and count_tokens(candidate) > budget:
            flush()
        pending.append(block)

    flush()
    return chunks


def chunk_corpus(
    documents: list[Document],
    config: ChunkingConfig | None = None,
    count_tokens: TokenCounter | None = None,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        document_chunks = chunk_document(document, config, count_tokens)
        logger.info("%-45s -> %3d chunks", document.doc_id, len(document_chunks))
        chunks.extend(document_chunks)
    return chunks

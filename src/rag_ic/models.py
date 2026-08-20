"""Domain models shared across ingestion, retrieval, generation and evaluation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SourceFormat = Literal["markdown", "pdf", "csv"]

#: Emitted by the model (and detected by the evaluation harness) when the retrieved
#: context does not contain the answer. Kept as a constant so the prompt, the parser
#: and the tests can never drift apart.
INSUFFICIENT_CONTEXT_MARKER = "CONTEXTO_INSUFICIENTE"


class Block(BaseModel):
    """A contiguous piece of source text with its position in the document.

    Loaders emit blocks; the chunker groups or splits them into :class:`Chunk`.
    """

    text: str
    section_path: tuple[str, ...] = ()
    page: int | None = None
    row: int | None = None
    #: Blocks that must not be merged with their neighbours (e.g. one CSV row).
    atomic: bool = False


class Document(BaseModel):
    """A loaded source document, before chunking."""

    doc_id: str
    title: str
    source_path: str
    source_format: SourceFormat
    blocks: list[Block]


class Chunk(BaseModel):
    """An indexed unit of retrieval."""

    chunk_id: str
    doc_id: str
    doc_title: str
    source_path: str
    source_format: SourceFormat
    section_path: tuple[str, ...] = ()
    text: str
    token_count: int
    ordinal: int
    page: int | None = None
    row: int | None = None

    @property
    def section_label(self) -> str:
        """Human-readable section, e.g. ``Política de EPI > 4. Fornecimento``."""
        return " > ".join(self.section_path) if self.section_path else "(documento)"

    @property
    def context_header(self) -> str:
        """Prefix prepended to the text so a chunk is self-contained once retrieved.

        Both the embedding input and the BM25 input use this, so a query naming a
        document or a section can match on that alone.
        """
        parts = [self.doc_title, *self.section_path]
        if self.page is not None:
            parts.append(f"página {self.page}")
        return " — ".join(parts)

    @property
    def indexable_text(self) -> str:
        """What actually gets embedded and lexically indexed."""
        return f"{self.context_header}\n{self.text}"

    @property
    def citation(self) -> str:
        """Reference string shown in the UI, e.g. ``Política de EPI § 4. Fornecimento``."""
        if not self.section_path:
            return self.doc_title
        return f"{self.doc_title} § {' > '.join(self.section_path)}"


class ScoredChunk(BaseModel):
    """A chunk with the scores from every retrieval channel that saw it."""

    chunk: Chunk
    dense_score: float | None = None
    bm25_score: float | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None
    fused_score: float = 0.0

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


class GoldReference(BaseModel):
    """Where the answer to a gold question actually lives."""

    doc_id: str
    #: Prefix match against the chunk's section path, joined with " > ".
    #: Empty string means "anywhere in this document".
    section: str = ""


class GoldQuestion(BaseModel):
    """One labelled evaluation question."""

    id: str
    question: str
    answerable: bool = True
    relevant: list[GoldReference] = Field(default_factory=list)
    notes: str = ""


class AnswerResult(BaseModel):
    """Full result of one question, used by the JSON endpoint and the evaluator."""

    question: str
    answer: str
    refused: bool
    sources: list[ScoredChunk]
    provider: str
    model: str
    strategy: str
    retrieval_ms: float
    generation_ms: float

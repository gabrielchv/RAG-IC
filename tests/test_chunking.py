"""Chunking is the part most likely to break silently, so it gets the most tests."""

from __future__ import annotations

from rag_ic.ingest.chunking import ChunkingConfig, chunk_document, split_long_text
from rag_ic.ingest.tokens import approximate_token_count
from rag_ic.models import Block, Document


def words(count: int, word: str = "palavra") -> str:
    return " ".join([word] * count)


def make_document(blocks: list[Block]) -> Document:
    return Document(
        doc_id="teste/doc",
        title="Documento de teste",
        source_path="teste/doc.md",
        source_format="markdown",
        blocks=blocks,
    )


def test_small_blocks_in_same_section_are_merged():
    document = make_document(
        [
            Block(text="Primeira frase.", section_path=("1. Seção",)),
            Block(text="Segunda frase.", section_path=("1. Seção",)),
        ]
    )
    chunks = chunk_document(document, ChunkingConfig(max_tokens=400))
    assert len(chunks) == 1
    assert "Primeira frase." in chunks[0].text
    assert "Segunda frase." in chunks[0].text


def test_section_change_forces_a_new_chunk():
    document = make_document(
        [
            Block(text="Conteúdo A.", section_path=("1. A",)),
            Block(text="Conteúdo B.", section_path=("2. B",)),
        ]
    )
    chunks = chunk_document(document)
    assert [chunk.section_path for chunk in chunks] == [("1. A",), ("2. B",)]


def test_atomic_blocks_are_never_merged():
    document = make_document(
        [
            Block(text="linha 1", section_path=("linha 1",), row=1, atomic=True),
            Block(text="linha 2", section_path=("linha 2",), row=2, atomic=True),
        ]
    )
    chunks = chunk_document(document)
    assert len(chunks) == 2
    assert chunks[0].row == 1
    assert chunks[1].row == 2


def test_long_block_is_split_within_budget():
    long_text = ". ".join(words(40) for _ in range(30)) + "."
    document = make_document([Block(text=long_text, section_path=("1. Longa",))])
    config = ChunkingConfig(max_tokens=200, header_reserve_tokens=40)

    chunks = chunk_document(document, config)

    assert len(chunks) > 1
    budget = config.max_tokens - config.header_reserve_tokens
    # A small tolerance: the last sentence is added before the budget check, so a
    # chunk may overshoot by at most one sentence.
    assert all(chunk.token_count <= budget * 1.5 for chunk in chunks)


def test_split_long_text_produces_overlap():
    sentences = [f"Frase numero {index} com algum conteudo adicional." for index in range(40)]
    text = " ".join(sentences)

    pieces = split_long_text(text, budget=60, overlap=20, count_tokens=approximate_token_count)

    assert len(pieces) > 1
    # Consecutive pieces must share at least one sentence.
    first_tail = pieces[0].split(". ")[-1].strip(". ")
    assert first_tail and first_tail in pieces[1]


def test_split_long_text_returns_input_when_it_fits():
    assert split_long_text("curto", 400, 60, approximate_token_count) == ["curto"]


def test_single_oversized_sentence_is_hard_split():
    # No sentence boundary at all: the fallback must still bound the piece size.
    text = words(500)
    pieces = split_long_text(text, budget=50, overlap=5, count_tokens=approximate_token_count)
    assert len(pieces) > 1
    assert all(approximate_token_count(piece) <= 60 for piece in pieces)


def test_chunk_ids_are_stable_and_ordered():
    document = make_document(
        [Block(text=f"Texto {index}.", section_path=(f"{index}. S",)) for index in range(3)]
    )
    chunks = chunk_document(document)
    assert [chunk.chunk_id for chunk in chunks] == [
        "teste/doc#0000",
        "teste/doc#0001",
        "teste/doc#0002",
    ]
    assert [chunk.ordinal for chunk in chunks] == [0, 1, 2]


def test_indexable_text_carries_the_context_header():
    document = make_document(
        [Block(text="A troca é semestral.", section_path=("4. Troca", "4.1 Prazo"))]
    )
    chunk = chunk_document(document)[0]
    assert chunk.indexable_text.startswith("Documento de teste — 4. Troca — 4.1 Prazo")
    assert "A troca é semestral." in chunk.indexable_text


def test_page_is_preserved_and_appears_in_the_header():
    document = Document(
        doc_id="pdf/doc",
        title="Comunicado",
        source_path="pdf/doc.pdf",
        source_format="pdf",
        blocks=[Block(text="Recolhimento do lote.", page=2)],
    )
    chunk = chunk_document(document)[0]
    assert chunk.page == 2
    assert "página 2" in chunk.context_header


def test_empty_document_yields_no_chunks():
    assert chunk_document(make_document([])) == []

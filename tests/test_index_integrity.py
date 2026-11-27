"""A saved index must not load if its two files disagree.

`save_index` writes the chunk file and the vector file separately, and
`load_index` used to check only that their *lengths* matched. A re-ingest that
changes chunk text while producing the same chunk count — a parser fix, a
reordered heading, a changed overlap that shifts boundaries without changing how
many there are — leaves the two files agreeing on length and disagreeing on
content. Retrieval then pairs every chunk with the wrong vector and returns
confident nonsense, with nothing anywhere saying so.
"""

from __future__ import annotations

import numpy as np
import pytest

from rag_ic.models import Chunk
from rag_ic.retrieval.store import load_index, save_index


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc",
        doc_title="Doc",
        source_path="doc.md",
        source_format="markdown",
        text=text,
        token_count=len(text.split()),
        ordinal=0,
    )


def write_index(directory, chunks: list[Chunk]) -> None:
    vectors = np.zeros((len(chunks), 4), dtype=np.float32)
    for row in range(len(chunks)):
        vectors[row, 0] = row
    save_index(
        directory,
        chunks,
        vectors,
        company="acme",
        embedding_model="test-model",
        max_chunk_tokens=100,
        chunk_overlap_ratio=0.0,
    )


def test_a_consistent_index_round_trips(tmp_path):
    write_index(tmp_path, [make_chunk("a", "primeiro"), make_chunk("b", "segundo")])

    loaded = load_index(tmp_path)

    assert [chunk.chunk_id for chunk in loaded.chunks] == ["a", "b"]
    assert loaded.vectors.shape == (2, 4)


def test_changed_chunk_text_with_the_same_count_is_refused(tmp_path):
    write_index(tmp_path, [make_chunk("a", "primeiro"), make_chunk("b", "segundo")])

    # Simulate a re-ingest that replaced the chunks but never rewrote vectors: same
    # count, different content, and nothing in the manifest to say so before this.
    (tmp_path / "chunks.jsonl").write_text(
        make_chunk("a", "texto trocado").model_dump_json()
        + "\n"
        + make_chunk("b", "outro texto").model_dump_json()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match the manifest digest"):
        load_index(tmp_path)


def test_a_truncated_chunk_file_is_still_caught(tmp_path):
    write_index(tmp_path, [make_chunk("a", "primeiro"), make_chunk("b", "segundo")])

    (tmp_path / "chunks.jsonl").write_text(
        make_chunk("a", "primeiro").model_dump_json() + "\n", encoding="utf-8"
    )

    # Count check fires first here, which is the clearer message.
    with pytest.raises(ValueError, match="chunk count does not match"):
        load_index(tmp_path)

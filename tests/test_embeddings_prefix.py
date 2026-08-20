"""e5 prefixes and the vector-store contract.

These are the two failure modes that produce a *working* system with silently bad
results: a missing ``passage: `` prefix, and a vector matrix that no longer lines up
with its chunk list.
"""

from __future__ import annotations

import numpy as np
import pytest

from rag_ic.embeddings.local_e5 import (
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    as_passage,
    as_query,
    resolve_device,
)
from rag_ic.models import Chunk
from rag_ic.retrieval.dense import DenseIndex
from rag_ic.retrieval.store import load_index, save_index


def test_query_prefix_is_applied():
    assert as_query("qual o CA da luva?") == f"{QUERY_PREFIX}qual o CA da luva?"


def test_passage_prefix_is_applied():
    assert as_passage("A troca é semestral.") == f"{PASSAGE_PREFIX}A troca é semestral."


def test_prefixes_are_not_doubled():
    once = as_query("pergunta")
    assert as_query(once) == once


def test_prefixes_are_distinct():
    # The whole point of e5 asymmetry: the same text embeds differently as a query
    # and as a passage.
    assert as_query("texto") != as_passage("texto")


def test_resolve_device_passes_explicit_values_through():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"


def test_resolve_device_auto_returns_something_usable():
    assert resolve_device("auto") in {"cpu", "cuda"}


def make_chunk(index: int) -> Chunk:
    return Chunk(
        chunk_id=f"doc#{index:04d}",
        doc_id="doc",
        doc_title="Doc",
        source_path="doc.md",
        source_format="markdown",
        section_path=("1. Seção",),
        text=f"texto {index}",
        token_count=3,
        ordinal=index,
    )


def test_index_round_trips(tmp_path):
    chunks = [make_chunk(index) for index in range(4)]
    vectors = np.random.default_rng(0).normal(size=(4, 8)).astype(np.float32)

    save_index(
        tmp_path,
        chunks,
        vectors,
        company="teste",
        embedding_model="fake-model",
        max_chunk_tokens=400,
        chunk_overlap_ratio=0.15,
    )
    loaded = load_index(tmp_path)

    assert [chunk.chunk_id for chunk in loaded.chunks] == [c.chunk_id for c in chunks]
    assert loaded.manifest.chunk_count == 4
    assert loaded.manifest.document_count == 1
    assert loaded.manifest.embedding_model == "fake-model"
    np.testing.assert_allclose(loaded.vectors, vectors, rtol=1e-6)


def test_saving_mismatched_counts_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="mismatch"):
        save_index(
            tmp_path,
            [make_chunk(0)],
            np.zeros((2, 8), dtype=np.float32),
            company="teste",
            embedding_model="fake",
            max_chunk_tokens=400,
            chunk_overlap_ratio=0.15,
        )


def test_loading_a_missing_index_explains_how_to_build_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_index"):
        load_index(tmp_path / "vazio")


def test_dense_search_ranks_by_cosine():
    vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]],
        dtype=np.float32,
    )
    hits = DenseIndex(vectors).search(np.array([1.0, 0.0], dtype=np.float32), top_k=3)

    assert [index for index, _ in hits] == [0, 2, 1]
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)


def test_dense_search_normalises_the_query():
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    scaled = DenseIndex(vectors).search(np.array([5.0, 0.0], dtype=np.float32), top_k=1)
    assert scaled[0][1] == pytest.approx(1.0, abs=1e-5)


def test_dense_search_on_empty_index_returns_nothing():
    assert DenseIndex(np.zeros((0, 8), dtype=np.float32)).search(np.zeros(8), top_k=5) == []

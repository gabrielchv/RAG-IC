"""`fused_score` must mean one thing.

It is the RRF score, which only exists when the ordering came from a fusion. The
retriever used to fill it with whatever ordering score it had: a cosine
similarity under `dense`, a BM25 score under `bm25`. Those are on incomparable
scales, so a field called `fused_score` carried a number that meant three
different things depending on a value elsewhere in the row.
"""

from __future__ import annotations

import numpy as np

from rag_ic.models import Chunk
from rag_ic.retrieval.retriever import HybridRetriever
from rag_ic.retrieval.store import IndexManifest, LoadedIndex


class StubEmbedder:
    """Constant query vector; the dense ranking is decided by the index vectors."""

    dimension = 2

    @property
    def name(self) -> str:
        return "stub"

    def embed_query(self, _text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


def chunk(index: int) -> Chunk:
    return Chunk(
        chunk_id=f"c{index}",
        doc_id="doc",
        doc_title="Doc",
        source_path="doc.md",
        source_format="markdown",
        text=f"texto do bloco numero {index} sobre equipamento",
        token_count=8,
        ordinal=index,
    )


def pipeline() -> HybridRetriever:
    chunks = [chunk(index) for index in range(3)]
    vectors = np.eye(3, 2, dtype=np.float32)
    manifest = IndexManifest(
        company="acme",
        embedding_model="stub",
        dimension=2,
        chunk_count=len(chunks),
        document_count=1,
        max_chunk_tokens=100,
        chunk_overlap_ratio=0.0,
        built_at="2026-01-01T00:00:00Z",
    )
    index = LoadedIndex(chunks=chunks, vectors=vectors, manifest=manifest)
    return HybridRetriever(index, StubEmbedder(), candidates_per_retriever=3, rrf_k=60)


def test_hybrid_produces_a_fused_score():
    results = pipeline().search("equipamento", top_k=3, strategy="hybrid")

    assert results
    assert all(result.fused_score is not None for result in results)


def test_dense_does_not_report_a_fused_score():
    results = pipeline().search("equipamento", top_k=3, strategy="dense")

    assert results
    assert all(result.fused_score is None for result in results)
    # The dense score is still reported, under its own name.
    assert any(result.dense_score is not None for result in results)


def test_bm25_does_not_report_a_fused_score():
    results = pipeline().search("equipamento", top_k=3, strategy="bm25")

    assert results
    assert all(result.fused_score is None for result in results)

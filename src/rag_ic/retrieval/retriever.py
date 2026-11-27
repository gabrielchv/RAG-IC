"""The retriever that the API and the evaluator both use.

One object owns the loaded index, the embedding model and the BM25 index, and can
answer with any of the three strategies. The evaluator's ablation table is exactly
this class called three times with a different ``strategy``.
"""

from __future__ import annotations

import logging

from rag_ic.config import RetrievalStrategy, Settings
from rag_ic.embeddings.base import EmbeddingProvider
from rag_ic.models import ScoredChunk
from rag_ic.retrieval.dense import DenseIndex
from rag_ic.retrieval.fusion import ranks_by_index, reciprocal_rank_fusion
from rag_ic.retrieval.lexical import BM25Index
from rag_ic.retrieval.store import LoadedIndex, load_index

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(
        self,
        index: LoadedIndex,
        embedder: EmbeddingProvider,
        *,
        candidates_per_retriever: int = 30,
        rrf_k: int = 60,
    ) -> None:
        self._index = index
        self._embedder = embedder
        self._dense = DenseIndex(index.vectors)
        self._bm25 = BM25Index(index.chunks)
        self._candidates = candidates_per_retriever
        self._rrf_k = rrf_k

        if index.manifest.embedding_model != embedder.name:
            logger.warning(
                "index was built with %s but %s is loaded — rebuild the index",
                index.manifest.embedding_model,
                embedder.name,
            )

    @classmethod
    def from_settings(cls, settings: Settings, embedder: EmbeddingProvider) -> HybridRetriever:
        return cls(
            load_index(settings.company_index_dir),
            embedder,
            candidates_per_retriever=settings.candidates_per_retriever,
            rrf_k=settings.rrf_k,
        )

    @property
    def index(self) -> LoadedIndex:
        return self._index

    def search(
        self,
        query: str,
        top_k: int = 5,
        strategy: RetrievalStrategy = "hybrid",
    ) -> list[ScoredChunk]:
        """Retrieve the ``top_k`` best chunks under the requested strategy.

        Both channels are always run, so the inspector can show a chunk's BM25 score
        even when the ranking came from dense alone. Only the *ordering* changes with
        the strategy.
        """
        chunks = self._index.chunks
        if not chunks:
            return []

        dense_hits = self._dense.search(self._embedder.embed_query(query), self._candidates)
        bm25_hits = self._bm25.search(query, self._candidates)

        dense_scores = dict(dense_hits)
        bm25_scores = dict(bm25_hits)
        dense_ranks = ranks_by_index([index for index, _ in dense_hits])
        bm25_ranks = ranks_by_index([index for index, _ in bm25_hits])

        # Only a fusion produces a fused score. Under `dense` or `bm25` the
        # ordering score is that retriever's own, on its own scale, and reporting
        # it as a fused score made one field mean three different things.
        fused_scores: dict[int, float] = {}

        if strategy == "dense":
            ordered: list[tuple[int, float]] = list(dense_hits)
        elif strategy == "bm25":
            ordered = list(bm25_hits)
        else:
            fused = reciprocal_rank_fusion(
                {
                    "dense": [index for index, _ in dense_hits],
                    "bm25": [index for index, _ in bm25_hits],
                },
                k=self._rrf_k,
            )
            ordered = sorted(fused.items(), key=lambda pair: pair[1], reverse=True)
            fused_scores = fused

        results = [
            ScoredChunk(
                chunk=chunks[index],
                dense_score=dense_scores.get(index),
                bm25_score=bm25_scores.get(index),
                dense_rank=dense_ranks.get(index),
                bm25_rank=bm25_ranks.get(index),
                fused_score=fused_scores.get(index),
            )
            for index, score in ordered[:top_k]
        ]
        logger.debug("query=%r strategy=%s -> %d chunks", query, strategy, len(results))
        return results

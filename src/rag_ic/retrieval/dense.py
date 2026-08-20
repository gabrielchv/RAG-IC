"""Exact dense retrieval over an in-memory NumPy matrix.

At a few thousand chunks, exact search is a single matrix multiplication costing well
under a millisecond — an approximate index (FAISS, HNSW) would add a dependency, a
build step and a recall/latency knob to tune, in exchange for nothing. That trade-off
flips somewhere around 10^5–10^6 vectors.

Vectors are stored L2-normalised, so cosine similarity *is* the dot product.
"""

from __future__ import annotations

import numpy as np


class DenseIndex:
    def __init__(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2:
            raise ValueError(f"expected a 2-D matrix, got shape {vectors.shape}")
        self._vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    @property
    def shape(self) -> tuple[int, int]:
        return self._vectors.shape  # type: ignore[return-value]

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Return ``(chunk_index, cosine_similarity)`` for the ``top_k`` nearest chunks."""
        if self._vectors.size == 0 or top_k <= 0:
            return []

        query = np.asarray(query_vector, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(query))
        if norm > 0:
            query = query / norm

        scores = self._vectors @ query
        top_k = min(top_k, scores.shape[0])
        # argpartition finds the top-k without sorting the whole array; sort only those.
        candidate_indices = np.argpartition(-scores, top_k - 1)[:top_k]
        ordered = candidate_indices[np.argsort(-scores[candidate_indices])]
        return [(int(index), float(scores[index])) for index in ordered]

"""Reciprocal Rank Fusion.

Cosine similarities live in [-1, 1] and cluster tightly; BM25 scores are unbounded
and corpus-dependent. Any attempt to add them directly requires normalising two
distributions that have nothing to do with each other, and the normalisation ends up
deciding the result.

RRF sidesteps this by throwing the scores away and keeping only the ranks::

    score(d) = Σ  1 / (k + rank_r(d))
               r

with ``k = 60`` from the original paper. The constant damps the difference between
the top positions, so a document ranked 1st by one retriever and absent from the
other does not automatically beat a document ranked 2nd and 3rd by both.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[int]],
    k: int = DEFAULT_RRF_K,
) -> dict[int, float]:
    """Fuse several ranked lists of chunk indices into one score per chunk.

    ``rankings`` maps a retriever name to its ranked list, best first. Ranks are
    1-based internally.
    """
    fused: dict[int, float] = {}
    for ranked_indices in rankings.values():
        for rank, index in enumerate(ranked_indices, start=1):
            fused[index] = fused.get(index, 0.0) + 1.0 / (k + rank)
    return fused


def ranks_by_index(ranked_indices: Sequence[int]) -> dict[int, int]:
    """1-based rank of each index in a ranked list, for display in the inspector."""
    return {index: rank for rank, index in enumerate(ranked_indices, start=1)}

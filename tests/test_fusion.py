"""Reciprocal Rank Fusion, checked against hand-computed values."""

from __future__ import annotations

import pytest

from rag_ic.retrieval.fusion import ranks_by_index, reciprocal_rank_fusion


def test_single_ranking_is_monotonically_decreasing():
    fused = reciprocal_rank_fusion({"dense": [10, 20, 30]}, k=60)
    assert fused[10] > fused[20] > fused[30]


def test_score_matches_the_formula():
    fused = reciprocal_rank_fusion({"dense": [7]}, k=60)
    assert fused[7] == pytest.approx(1 / 61)


def test_agreement_between_retrievers_beats_a_single_first_place():
    # Chunk 2 is 2nd and 2nd; chunk 1 is 1st and absent.
    # 2/62 = 0.03226 > 1/61 = 0.01639
    fused = reciprocal_rank_fusion({"dense": [1, 2], "bm25": [3, 2]}, k=60)
    assert fused[2] > fused[1]
    assert fused[2] == pytest.approx(2 / 62)


def test_smaller_k_sharpens_the_advantage_of_rank_one():
    sharp = reciprocal_rank_fusion({"dense": [1, 2]}, k=1)
    flat = reciprocal_rank_fusion({"dense": [1, 2]}, k=1000)
    assert sharp[1] / sharp[2] > flat[1] / flat[2]


def test_empty_rankings_produce_no_scores():
    assert reciprocal_rank_fusion({"dense": [], "bm25": []}) == {}


def test_ranks_by_index_is_one_based():
    assert ranks_by_index([5, 9, 2]) == {5: 1, 9: 2, 2: 3}

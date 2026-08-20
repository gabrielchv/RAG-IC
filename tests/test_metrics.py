"""Metrics produce the numbers that go in the paper, so they get tested first."""

from __future__ import annotations

import pytest

from rag_ic.evaluation.metrics import (
    RefusalReport,
    evaluate_retrieval,
    hit_at_k,
    matches_reference,
    recall_at_k,
    reciprocal_rank,
)
from rag_ic.models import Chunk, GoldQuestion, GoldReference, ScoredChunk


def chunk(doc_id: str, section: tuple[str, ...] = ()) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}#{'-'.join(section)}",
        doc_id=doc_id,
        doc_title=doc_id,
        source_path=f"{doc_id}.md",
        source_format="markdown",
        section_path=section,
        text="texto",
        token_count=1,
        ordinal=0,
    )


def scored(doc_id: str, section: tuple[str, ...] = ()) -> ScoredChunk:
    return ScoredChunk(chunk=chunk(doc_id, section))


def question(*references: GoldReference, answerable: bool = True) -> GoldQuestion:
    return GoldQuestion(
        id="Q", question="pergunta", answerable=answerable, relevant=list(references)
    )


def test_reference_with_empty_section_matches_any_chunk_of_the_document():
    assert matches_reference(
        chunk("politica-epi", ("7. Penalidades",)), GoldReference(doc_id="politica-epi")
    )


def test_reference_section_is_a_prefix_match():
    reference = GoldReference(doc_id="d", section="1. Capacete")
    assert matches_reference(chunk("d", ("1. Capacete", "1.2 Vida útil")), reference)
    assert not matches_reference(chunk("d", ("2. Calçado",)), reference)


def test_wrong_document_never_matches():
    assert not matches_reference(chunk("outro"), GoldReference(doc_id="d"))


def test_recall_counts_regions_not_chunks():
    gold = question(GoldReference(doc_id="a"), GoldReference(doc_id="b"))
    retrieved = [scored("a"), scored("a"), scored("c")]
    assert recall_at_k(retrieved, gold, k=3) == pytest.approx(0.5)


def test_recall_reaches_one_when_every_region_is_hit():
    gold = question(GoldReference(doc_id="a"), GoldReference(doc_id="b"))
    assert recall_at_k([scored("a"), scored("b")], gold, k=2) == pytest.approx(1.0)


def test_recall_respects_the_cutoff():
    gold = question(GoldReference(doc_id="b"))
    retrieved = [scored("a"), scored("b")]
    assert recall_at_k(retrieved, gold, k=1) == 0.0
    assert recall_at_k(retrieved, gold, k=2) == 1.0


def test_hit_at_k_is_lenient_where_recall_is_strict():
    gold = question(GoldReference(doc_id="a"), GoldReference(doc_id="b"))
    retrieved = [scored("a")]
    assert hit_at_k(retrieved, gold, k=1) == 1.0
    assert recall_at_k(retrieved, gold, k=1) == pytest.approx(0.5)


def test_reciprocal_rank_uses_the_first_relevant_position():
    gold = question(GoldReference(doc_id="b"))
    assert reciprocal_rank([scored("a"), scored("b"), scored("c")], gold) == pytest.approx(1 / 2)


def test_reciprocal_rank_is_zero_without_any_hit():
    assert reciprocal_rank([scored("x")], question(GoldReference(doc_id="b"))) == 0.0


def test_evaluate_retrieval_aggregates_and_lists_misses():
    questions = [
        GoldQuestion(id="Q1", question="a", relevant=[GoldReference(doc_id="a")]),
        GoldQuestion(id="Q2", question="b", relevant=[GoldReference(doc_id="b")]),
    ]
    results = {"Q1": [scored("a")], "Q2": [scored("z")]}

    report = evaluate_retrieval("hybrid", results, questions, ks=(1,))

    assert report.question_count == 2
    assert report.recall[1] == pytest.approx(0.5)
    assert report.mrr == pytest.approx(0.5)
    assert report.misses == ["Q2"]


def test_unanswerable_questions_are_excluded_from_retrieval_metrics():
    questions = [
        GoldQuestion(id="Q1", question="a", relevant=[GoldReference(doc_id="a")]),
        GoldQuestion(id="U1", question="u", answerable=False),
    ]
    report = evaluate_retrieval("dense", {"Q1": [scored("a")]}, questions, ks=(1,))
    assert report.question_count == 1
    assert report.recall[1] == pytest.approx(1.0)


def test_refusal_report_rates():
    report = RefusalReport(
        answerable_total=50,
        answerable_refused=5,
        unanswerable_total=10,
        unanswerable_refused=8,
    )
    assert report.false_refusal_rate == pytest.approx(0.1)
    assert report.true_negative_rate == pytest.approx(0.8)
    assert report.accuracy == pytest.approx((45 + 8) / 60)

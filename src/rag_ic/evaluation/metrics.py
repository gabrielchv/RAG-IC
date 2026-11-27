"""Retrieval and refusal metrics.

Relevance is judged against ``(doc_id, section)`` labels rather than chunk ids on
purpose. Chunk ids shift whenever the chunk size, the overlap or the loader changes,
and a gold set that has to be relabelled after every chunking experiment is a gold
set nobody will maintain. Document and section survive those changes, so the same
labels support the whole ablation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from rag_ic.models import Chunk, GoldQuestion, GoldReference, ScoredChunk


def matches_reference(chunk: Chunk, reference: GoldReference) -> bool:
    """Whether ``chunk`` lies inside the region named by ``reference``.

    An empty ``section`` means "anywhere in this document" — used for PDFs and CSVs,
    whose chunks carry pages and rows rather than headings.

    A non-empty ``section`` is matched on a component boundary, not as a bare
    prefix. ``"40. Disposições finais".startswith("4")`` is true, so a gold label
    of ``"4"`` used to accept a chunk from section 40 and credit a section the
    question was never about. Sections are joined with ``" > "``, so a match is
    either the whole path or the path followed by that separator.
    """
    if chunk.doc_id != reference.doc_id:
        return False
    if not reference.section:
        return True
    path = " > ".join(chunk.section_path)
    return path == reference.section or path.startswith(reference.section + " > ")


def is_relevant(chunk: Chunk, references: Sequence[GoldReference]) -> bool:
    return any(matches_reference(chunk, reference) for reference in references)


def recall_at_k(retrieved: Sequence[ScoredChunk], question: GoldQuestion, k: int) -> float:
    """Fraction of the question's labelled regions hit within the top ``k``.

    Regions, not chunks: a question labelled with two documents scores 0.5 when only
    one of them appears. This is stricter than "any hit counts" and is what makes the
    multi-document questions in the gold set worth having.
    """
    if not question.relevant:
        return 0.0
    top = [scored.chunk for scored in retrieved[:k]]
    hits = sum(
        1
        for reference in question.relevant
        if any(matches_reference(chunk, reference) for chunk in top)
    )
    return hits / len(question.relevant)


def hit_at_k(retrieved: Sequence[ScoredChunk], question: GoldQuestion, k: int) -> float:
    """1.0 if any labelled region appears in the top ``k``. The lenient counterpart."""
    return float(any(is_relevant(scored.chunk, question.relevant) for scored in retrieved[:k]))


def reciprocal_rank(retrieved: Sequence[ScoredChunk], question: GoldQuestion) -> float:
    """1 / rank of the first relevant chunk, or 0 if none was retrieved."""
    for rank, scored in enumerate(retrieved, start=1):
        if is_relevant(scored.chunk, question.relevant):
            return 1.0 / rank
    return 0.0


@dataclass
class RetrievalReport:
    """Aggregate over the answerable questions of one strategy."""

    strategy: str
    question_count: int
    mrr: float
    recall: dict[int, float] = field(default_factory=dict)
    hit_rate: dict[int, float] = field(default_factory=dict)
    misses: list[str] = field(default_factory=list)

    def as_row(self, ks: Sequence[int]) -> list[str]:
        return [
            self.strategy,
            f"{self.mrr:.3f}",
            *[f"{self.recall[k]:.3f}" for k in ks],
            *[f"{self.hit_rate[k]:.3f}" for k in ks],
            str(len(self.misses)),
        ]


def evaluate_retrieval(
    strategy: str,
    results: dict[str, list[ScoredChunk]],
    questions: Sequence[GoldQuestion],
    ks: Sequence[int] = (1, 3, 5, 10),
) -> RetrievalReport:
    """Aggregate per-question retrievals into one report.

    ``results`` maps question id to the ranked chunks retrieved for it.
    """
    answerable = [question for question in questions if question.answerable]
    if not answerable:
        raise ValueError("no answerable questions to evaluate")

    report = RetrievalReport(
        strategy=strategy,
        question_count=len(answerable),
        mrr=0.0,
    )
    for k in ks:
        report.recall[k] = sum(recall_at_k(results[q.id], q, k) for q in answerable) / len(
            answerable
        )
        report.hit_rate[k] = sum(hit_at_k(results[q.id], q, k) for q in answerable) / len(
            answerable
        )

    report.mrr = sum(reciprocal_rank(results[q.id], q) for q in answerable) / len(answerable)
    report.misses = [q.id for q in answerable if hit_at_k(results[q.id], q, max(ks)) == 0.0]
    return report


@dataclass
class RefusalReport:
    """How well the model distinguishes answerable from unanswerable questions."""

    answerable_total: int
    answerable_refused: int
    unanswerable_total: int
    unanswerable_refused: int
    invalid_citations: list[str] = field(default_factory=list)

    @property
    def true_negative_rate(self) -> float:
        """Share of unanswerable questions correctly refused. Higher is better."""
        if not self.unanswerable_total:
            return 0.0
        return self.unanswerable_refused / self.unanswerable_total

    @property
    def false_refusal_rate(self) -> float:
        """Share of answerable questions wrongly refused. Lower is better."""
        if not self.answerable_total:
            return 0.0
        return self.answerable_refused / self.answerable_total

    @property
    def accuracy(self) -> float:
        total = self.answerable_total + self.unanswerable_total
        if not total:
            return 0.0
        correct = (self.answerable_total - self.answerable_refused) + self.unanswerable_refused
        return correct / total

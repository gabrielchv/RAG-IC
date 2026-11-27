"""The ablation must search deep enough for the cutoffs it reports.

`run_retrieval_ablation` retrieves with `top_k` and then scores R@1/3/5/10. If the
requested depth is shallower than the largest reported cutoff, the metric is
capped by the flag rather than measured: R@10 over a 3-deep retrieval cannot
exceed 0.3 however good the retriever is.
"""

from __future__ import annotations

from rag_ic.evaluation.run_eval import KS, run_retrieval_ablation
from rag_ic.models import GoldQuestion, GoldReference


class RecordingPipeline:
    """Records the depth it was asked for, and returns that many chunks."""

    def __init__(self) -> None:
        self.depths: list[int] = []

    def retrieve(self, _question: str, *, top_k: int, strategy: str):
        self.depths.append(top_k)
        return []


def _questions() -> list[GoldQuestion]:
    return [
        GoldQuestion(
            id="Q1",
            question="pergunta",
            answerable=True,
            relevant=[GoldReference(doc_id="d", section="1")],
        )
    ]


def test_retrieval_depth_covers_the_largest_reported_cutoff():
    pipeline = RecordingPipeline()

    run_retrieval_ablation(pipeline, _questions(), top_k=3)

    assert pipeline.depths, "the pipeline was never asked for anything"
    assert all(depth >= max(KS) for depth in pipeline.depths), pipeline.depths


def test_a_deeper_explicit_request_is_honoured():
    pipeline = RecordingPipeline()

    run_retrieval_ablation(pipeline, _questions(), top_k=max(KS) + 5)

    assert all(depth >= max(KS) + 5 for depth in pipeline.depths), pipeline.depths

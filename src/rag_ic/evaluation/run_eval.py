"""Run the retrieval ablation (and, optionally, the generation evaluation).

    python -m rag_ic.evaluation.run_eval                # retrieval only, free
    python -m rag_ic.evaluation.run_eval --with-llm     # also measures refusal

The retrieval pass costs nothing and is deterministic: the same index and the same
questions always produce the same table. The ``--with-llm`` pass spends tokens and is
not deterministic, so it is opt-in and reported separately.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import re
from pathlib import Path

from rag_ic.config import PROJECT_ROOT, Settings, get_settings
from rag_ic.evaluation.metrics import (
    RefusalReport,
    RetrievalReport,
    evaluate_retrieval,
)
from rag_ic.logging_setup import configure_logging
from rag_ic.models import GoldQuestion
from rag_ic.pipeline import RagPipeline, build_pipeline

logger = logging.getLogger(__name__)

STRATEGIES = ("dense", "bm25", "hybrid")
KS = (1, 3, 5, 10)
CITATION_RE = re.compile(r"\[(\d{1,2})\]")


def load_gold(path: Path) -> list[GoldQuestion]:
    questions = [
        GoldQuestion.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not questions:
        raise SystemExit(f"gold set is empty: {path}")
    return questions


def run_retrieval_ablation(
    pipeline: RagPipeline,
    questions: list[GoldQuestion],
    top_k: int,
) -> list[RetrievalReport]:
    """Score every strategy at every cutoff in `KS`.

    The retrieval depth is `max(top_k, max(KS))`, not `top_k`. Scoring R@10 against
    a `--top-k 3` retrieval silently caps recall at 0.3 no matter how good the
    system is: there are only three candidates to find anything in. A cutoff a
    report prints has to actually have been searched to that depth, or the number
    is a property of the flag rather than of the retriever (and the report then
    under-reports the system it is measuring).
    """
    answerable = [question for question in questions if question.answerable]
    depth = max(top_k, max(KS))
    reports = []
    for strategy in STRATEGIES:
        results = {
            question.id: pipeline.retrieve(question.question, top_k=depth, strategy=strategy)
            for question in answerable
        }
        reports.append(evaluate_retrieval(strategy, results, answerable, KS))
    return reports


async def run_generation_eval(
    pipeline: RagPipeline,
    questions: list[GoldQuestion],
) -> tuple[RefusalReport, list[dict[str, str]]]:
    report = RefusalReport(
        answerable_total=sum(1 for q in questions if q.answerable),
        answerable_refused=0,
        unanswerable_total=sum(1 for q in questions if not q.answerable),
        unanswerable_refused=0,
    )
    rows: list[dict[str, str]] = []

    for question in questions:
        result = await pipeline.answer(question.question)
        if result.refused:
            if question.answerable:
                report.answerable_refused += 1
            else:
                report.unanswerable_refused += 1

        # A citation pointing past the number of sources shown to the model is a
        # fabricated reference — cheap to detect, and the clearest grounding failure.
        cited = {int(number) for number in CITATION_RE.findall(result.answer)}
        source_count = len(result.sources)
        if any(number < 1 or number > source_count for number in cited):
            report.invalid_citations.append(question.id)

        rows.append(
            {
                "id": question.id,
                "answerable": str(question.answerable),
                "refused": str(result.refused),
                "citations": " ".join(str(number) for number in sorted(cited)),
                "question": question.question,
                "answer": result.answer.replace("\n", " "),
            }
        )
        logger.info(
            "%s answerable=%s refused=%s citations=%d",
            question.id,
            question.answerable,
            result.refused,
            len(cited),
        )

    return report, rows


def print_table(reports: list[RetrievalReport]) -> None:
    header = [
        "estratégia",
        "MRR",
        *[f"R@{k}" for k in KS],
        *[f"Hit@{k}" for k in KS],
        "falhas",
    ]
    rows = [report.as_row(KS) for report in reports]
    widths = [
        max(len(header[column]), *(len(row[column]) for row in rows))
        for column in range(len(header))
    ]

    def line(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(cells))

    print()
    print(line(header))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(line(row))
    print()

    for report in reports:
        if report.misses:
            print(
                f"{report.strategy}: nenhuma passagem relevante em top-{max(KS)} para "
                f"{', '.join(report.misses)}"
            )


def write_csv(path: Path, reports: list[RetrievalReport]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "strategy",
                "questions",
                "mrr",
                *[f"recall@{k}" for k in KS],
                *[f"hit@{k}" for k in KS],
                "misses",
            ]
        )
        for report in reports:
            writer.writerow(
                [
                    report.strategy,
                    report.question_count,
                    f"{report.mrr:.4f}",
                    *[f"{report.recall[k]:.4f}" for k in KS],
                    *[f"{report.hit_rate[k]:.4f}" for k in KS],
                    " ".join(report.misses),
                ]
            )
    print(f"tabela salva em {path}")


def write_answers_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["id", "answerable", "refused", "citations", "question", "answer"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"respostas salvas em {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company", help="overrides RAG_COMPANY")
    parser.add_argument(
        "--top-k",
        type=int,
        default=max(KS),
        help=(
            "retrieval depth; raised to max(KS) when smaller, so every reported "
            "recall@k was actually searched to depth k"
        ),
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="também avalia recusa e citações (gasta tokens do Gemini)",
    )
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "reports")
    args = parser.parse_args()

    settings: Settings = get_settings()
    if args.company:
        settings = settings.model_copy(update={"company": args.company})
    configure_logging(settings.log_level)

    questions = load_gold(settings.company_gold_dir / "questions.jsonl")
    pipeline = build_pipeline(settings, with_llm=args.with_llm)

    print(
        f"\n{len(questions)} perguntas "
        f"({sum(q.answerable for q in questions)} respondíveis, "
        f"{sum(not q.answerable for q in questions)} sem resposta no acervo)"
    )

    reports = run_retrieval_ablation(pipeline, questions, args.top_k)
    print_table(reports)
    write_csv(args.out / f"retrieval_{settings.company}.csv", reports)

    if not args.with_llm:
        print("\n(recusa e citações não avaliadas — use --with-llm)")
        return

    refusal, rows = asyncio.run(run_generation_eval(pipeline, questions))
    print()
    print(
        f"recusa correta em perguntas sem resposta: {refusal.true_negative_rate:.1%} "
        f"({refusal.unanswerable_refused}/{refusal.unanswerable_total})"
    )
    print(
        f"recusa indevida em perguntas respondíveis: {refusal.false_refusal_rate:.1%} "
        f"({refusal.answerable_refused}/{refusal.answerable_total})"
    )
    print(f"acurácia global de recusa: {refusal.accuracy:.1%}")
    if refusal.invalid_citations:
        print(f"citações inválidas em: {', '.join(refusal.invalid_citations)}")
    else:
        print("nenhuma citação apontou para trecho inexistente")
    write_answers_csv(args.out / f"answers_{settings.company}.csv", rows)


if __name__ == "__main__":
    main()

"""The RAG pipeline: retrieve, then generate against what was retrieved.

Retrieval and generation are deliberately separable. ``retrieve`` is used on its own
by the evaluation harness, which measures Recall@k and MRR without spending a single
token — the ablation over dense / BM25 / hybrid is free and deterministic.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Any, Literal

from pydantic import BaseModel

from rag_ic.config import RetrievalStrategy, Settings
from rag_ic.embeddings.base import EmbeddingProvider
from rag_ic.embeddings.local_e5 import E5EmbeddingProvider
from rag_ic.llm.base import LLMProvider
from rag_ic.models import AnswerResult, ScoredChunk
from rag_ic.prompts import build_system_prompt, build_user_prompt, is_refusal
from rag_ic.retrieval.retriever import HybridRetriever

logger = logging.getLogger(__name__)


class StreamEvent(BaseModel):
    """One Server-Sent Event. ``sources`` first, then ``delta``s, then ``done``."""

    #: "failure" rather than "error": EventSource already dispatches a built-in
    #: `error` event for transport problems, and overloading the name makes the two
    #: indistinguishable in the browser.
    event: Literal["sources", "delta", "done", "failure"]
    data: dict[str, Any]


class RagPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLMProvider | None,
        settings: Settings,
    ) -> None:
        self._retriever = retriever
        self._maybe_llm = llm
        self._settings = settings
        self._system_prompt = build_system_prompt(settings.company_display_name)

    @property
    def llm(self) -> LLMProvider:
        """The generation provider.

        ``None`` is a legal construction state: the retrieval ablation runs without a
        provider (and therefore without an API key). Asking for one anyway is an error.
        """
        if self._maybe_llm is None:
            raise RuntimeError("this pipeline was built without an LLM provider (retrieval only)")
        return self._maybe_llm

    @property
    def retriever(self) -> HybridRetriever:
        return self._retriever

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        strategy: RetrievalStrategy | None = None,
    ) -> list[ScoredChunk]:
        return self._retriever.search(
            question,
            top_k=top_k or self._settings.top_k,
            strategy=strategy or self._settings.retrieval_strategy,
        )

    async def answer(
        self,
        question: str,
        top_k: int | None = None,
        strategy: RetrievalStrategy | None = None,
    ) -> AnswerResult:
        """Non-streaming path, used by ``POST /api/ask`` and the evaluator."""
        strategy = strategy or self._settings.retrieval_strategy

        started = time.perf_counter()
        sources = self.retrieve(question, top_k, strategy)
        retrieval_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        text = await self.llm.generate(self._system_prompt, build_user_prompt(question, sources))
        generation_ms = (time.perf_counter() - started) * 1000

        return AnswerResult(
            question=question,
            answer=text.strip(),
            refused=is_refusal(text),
            sources=sources,
            provider=self.llm.name,
            model=self.llm.model,
            strategy=strategy,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
        )

    async def answer_stream(
        self,
        question: str,
        top_k: int | None = None,
        strategy: RetrievalStrategy | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Streaming path. Sources are emitted before generation starts, so the
        inspector panel is populated while the answer is still being written."""
        strategy = strategy or self._settings.retrieval_strategy

        started = time.perf_counter()
        try:
            sources = self.retrieve(question, top_k, strategy)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("retrieval failed")
            yield StreamEvent(event="failure", data={"message": f"falha na recuperação: {exc}"})
            return
        retrieval_ms = (time.perf_counter() - started) * 1000

        yield StreamEvent(
            event="sources",
            data={
                "strategy": strategy,
                "retrieval_ms": round(retrieval_ms, 1),
                "sources": [scored.model_dump(mode="json") for scored in sources],
            },
        )

        prompt = build_user_prompt(question, sources)
        collected: list[str] = []
        started = time.perf_counter()
        try:
            async for delta in self.llm.generate_stream(self._system_prompt, prompt):
                collected.append(delta)
                yield StreamEvent(event="delta", data={"text": delta})
        except Exception as exc:
            logger.exception("generation failed")
            yield StreamEvent(event="failure", data={"message": f"falha na geração: {exc}"})
            return

        answer = "".join(collected)
        yield StreamEvent(
            event="done",
            data={
                "refused": is_refusal(answer),
                "generation_ms": round((time.perf_counter() - started) * 1000, 1),
                "provider": self.llm.name,
                "model": self.llm.model,
                "chars": len(answer),
            },
        )


def build_embedder(settings: Settings) -> EmbeddingProvider:
    return E5EmbeddingProvider(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )


def build_llm(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "gemini":
        from rag_ic.llm.gemini import GeminiProvider

        return GeminiProvider.from_settings(settings)
    if settings.llm_provider == "ollama":
        from rag_ic.llm.ollama import OllamaProvider

        return OllamaProvider.from_settings(settings)
    raise ValueError(f"unknown LLM provider: {settings.llm_provider}")


def build_pipeline(
    settings: Settings,
    embedder: EmbeddingProvider | None = None,
    *,
    with_llm: bool = True,
) -> RagPipeline:
    """Load the index and the models once. Called from the API lifespan and the CLI.

    ``with_llm=False`` skips provider construction entirely, so retrieval-only work
    (the ablation) runs with no API key configured.
    """
    embedder = embedder or build_embedder(settings)
    retriever = HybridRetriever.from_settings(settings, embedder)
    logger.info(
        "índice carregado: %d chunks de %d documentos (%s)",
        retriever.index.manifest.chunk_count,
        retriever.index.manifest.document_count,
        retriever.index.manifest.embedding_model,
    )
    return RagPipeline(retriever, build_llm(settings) if with_llm else None, settings)

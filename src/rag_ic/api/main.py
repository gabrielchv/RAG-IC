"""FastAPI application.

Run with::

    uvicorn rag_ic.api.main:app --reload

The index and the embedding model are loaded once in the lifespan handler, not per
request — loading e5-small takes a couple of seconds and rebuilding the BM25 index
per request would dominate the latency budget.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag_ic.config import RetrievalStrategy, Settings, get_settings
from rag_ic.logging_setup import configure_logging
from rag_ic.models import AnswerResult, ScoredChunk
from rag_ic.pipeline import RagPipeline, build_pipeline

logger = logging.getLogger(__name__)

MAX_QUESTION_LENGTH = 500


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=MAX_QUESTION_LENGTH)
    top_k: int | None = Field(default=None, ge=1, le=20)
    strategy: RetrievalStrategy | None = None


class RetrieveResponse(BaseModel):
    question: str
    strategy: str
    sources: list[ScoredChunk]


class ConfigResponse(BaseModel):
    company: str
    company_display_name: str
    embedding_model: str
    llm_provider: str
    llm_model: str
    llm_is_local: bool
    strategy: str
    top_k: int
    chunk_count: int
    document_count: int
    index_built_at: str


class AppState:
    """Holds the single loaded pipeline; ``None`` until startup finishes."""

    pipeline: RagPipeline | None = None
    settings: Settings | None = None
    startup_error: str | None = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    state.settings = settings
    try:
        state.pipeline = build_pipeline(settings)
    except Exception as exc:
        # Starting anyway means /api/health can explain the problem in the browser
        # instead of the server dying with a traceback the user never sees.
        state.startup_error = str(exc)
        logger.error("falha ao iniciar o pipeline: %s", exc)
    yield
    state.pipeline = None


app = FastAPI(
    title="RAG-IC — Assistente interno de EPI",
    description="Proof of concept de RAG local para perguntas internas sobre EPI.",
    version="0.1.0",
    lifespan=lifespan,
)


def require_pipeline() -> RagPipeline:
    if state.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=state.startup_error or "pipeline não inicializado",
        )
    return state.pipeline


@app.get("/api/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok" if state.pipeline is not None else "degraded",
        "error": state.startup_error,
    }


@app.get("/api/config", response_model=ConfigResponse)
async def config() -> ConfigResponse:
    pipeline = require_pipeline()
    settings = state.settings
    assert settings is not None
    manifest = pipeline.retriever.index.manifest
    return ConfigResponse(
        company=settings.company,
        company_display_name=settings.company_display_name,
        embedding_model=manifest.embedding_model,
        llm_provider=pipeline.llm.name,
        llm_model=pipeline.llm.model,
        llm_is_local=pipeline.llm.is_local,
        strategy=settings.retrieval_strategy,
        top_k=settings.top_k,
        chunk_count=manifest.chunk_count,
        document_count=manifest.document_count,
        index_built_at=manifest.built_at,
    )


@app.post("/api/ask", response_model=AnswerResult)
async def ask(request: AskRequest) -> AnswerResult:
    """Blocking variant: one JSON response with the answer and its sources."""
    pipeline = require_pipeline()
    return await pipeline.answer(request.question, request.top_k, request.strategy)


@app.get("/api/retrieve", response_model=RetrieveResponse)
async def retrieve(
    q: str = Query(min_length=3, max_length=MAX_QUESTION_LENGTH),
    top_k: int | None = Query(default=None, ge=1, le=20),
    strategy: RetrievalStrategy | None = None,
) -> RetrieveResponse:
    """Retrieval only — no tokens spent. Useful for inspecting the index by hand."""
    pipeline = require_pipeline()
    settings = state.settings
    assert settings is not None
    sources = pipeline.retrieve(q, top_k, strategy)
    return RetrieveResponse(
        question=q,
        strategy=strategy or settings.retrieval_strategy,
        sources=sources,
    )


def _sse(event: str, data: dict[str, object]) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


@app.get("/api/ask/stream")
async def ask_stream(
    q: str = Query(min_length=3, max_length=MAX_QUESTION_LENGTH),
    top_k: int | None = Query(default=None, ge=1, le=20),
    strategy: RetrievalStrategy | None = None,
) -> StreamingResponse:
    """Server-Sent Events. GET because that is what ``EventSource`` speaks."""
    pipeline = require_pipeline()

    async def event_stream() -> AsyncIterator[str]:
        async for event in pipeline.answer_stream(q, top_k, strategy):
            yield _sse(event.event, event.data)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


_settings_for_static = get_settings()
if _settings_for_static.web_dir.is_dir():
    app.mount(
        "/static",
        StaticFiles(directory=_settings_for_static.web_dir),
        name="static",
    )

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(_settings_for_static.web_dir / "index.html")

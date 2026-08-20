"""CLI: build the index for one company.

    python -m rag_ic.ingest.build_index --company aurora

Everything runs locally. Nothing in this command touches the network except the
first-run download of the embedding model weights.
"""

from __future__ import annotations

import argparse
import logging
import time

from rag_ic.config import Settings, get_settings
from rag_ic.embeddings.local_e5 import E5EmbeddingProvider
from rag_ic.ingest.chunking import ChunkingConfig, chunk_corpus
from rag_ic.ingest.loaders import load_corpus
from rag_ic.ingest.tokens import transformer_token_counter
from rag_ic.logging_setup import configure_logging
from rag_ic.retrieval.store import save_index

logger = logging.getLogger(__name__)


def build(settings: Settings) -> None:
    started = time.perf_counter()

    documents = load_corpus(settings.company_docs_dir)
    if not documents:
        raise SystemExit(f"no documents found in {settings.company_docs_dir}")

    count_tokens = transformer_token_counter(settings.embedding_model)
    chunks = chunk_corpus(
        documents,
        ChunkingConfig(
            max_tokens=settings.max_chunk_tokens,
            overlap_ratio=settings.chunk_overlap_ratio,
        ),
        count_tokens,
    )
    if not chunks:
        raise SystemExit("corpus produced no chunks")

    embedder = E5EmbeddingProvider(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
    vectors = embedder.embed_documents([chunk.indexable_text for chunk in chunks])

    manifest = save_index(
        settings.company_index_dir,
        chunks,
        vectors,
        company=settings.company,
        embedding_model=embedder.name,
        max_chunk_tokens=settings.max_chunk_tokens,
        chunk_overlap_ratio=settings.chunk_overlap_ratio,
    )

    token_counts = sorted(chunk.token_count for chunk in chunks)
    median = token_counts[len(token_counts) // 2]
    logger.info(
        "%d documentos, %d chunks (mediana %d tokens, máx %d), dim=%d, %.1fs",
        manifest.document_count,
        manifest.chunk_count,
        median,
        token_counts[-1],
        manifest.dimension,
        time.perf_counter() - started,
    )
    over_limit = [chunk for chunk in chunks if chunk.token_count > 512]
    if over_limit:
        logger.warning(
            "%d chunks excedem 512 tokens e serão truncados pelo encoder: %s",
            len(over_limit),
            ", ".join(chunk.chunk_id for chunk in over_limit[:5]),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local RAG index for one company.")
    parser.add_argument("--company", help="overrides RAG_COMPANY")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args()

    settings = get_settings()
    if args.company:
        settings = settings.model_copy(update={"company": args.company})
    configure_logging(args.log_level or settings.log_level)
    build(settings)


if __name__ == "__main__":
    main()

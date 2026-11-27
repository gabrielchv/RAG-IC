"""On-disk index format.

Three files per company, all human-inspectable except the vectors:

``chunks.jsonl``   one chunk per line, in vector-row order
``vectors.npz``    float32 matrix, L2-normalised, shape ``(n_chunks, dimension)``
``manifest.json``  what built it, so a stale index can be detected instead of trusted

The ordering contract — row *i* of the matrix is line *i* of the JSONL — is the whole
data structure. It is asserted on load rather than assumed.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from rag_ic.models import Chunk

logger = logging.getLogger(__name__)

INDEX_VERSION = 1

CHUNKS_FILE = "chunks.jsonl"
VECTORS_FILE = "vectors.npz"
MANIFEST_FILE = "manifest.json"


class IndexManifest(BaseModel):
    version: int = INDEX_VERSION
    company: str
    embedding_model: str
    dimension: int
    chunk_count: int
    document_count: int
    max_chunk_tokens: int
    chunk_overlap_ratio: float
    index_chunks_sha256: str = ""
    """Digest of the chunk file, so a half-updated index is detectable.

    `load_index` already checks that the chunk count matches the vector rows, but
    that only catches a truncated write. A re-ingest that changes chunk *text*
    while producing the same *count* — a parser fix, a reordering, a changed
    heading — leaves the two files agreeing on length and disagreeing on content,
    and retrieval then pairs every chunk with the wrong vector without saying
    anything. The digest makes the pairing verifiable."""
    built_at: str


class LoadedIndex(BaseModel):
    chunks: list[Chunk]
    manifest: IndexManifest
    #: Excluded from pydantic validation; assigned after construction.
    model_config = {"arbitrary_types_allowed": True}

    vectors: np.ndarray


def save_index(
    directory: Path,
    chunks: list[Chunk],
    vectors: np.ndarray,
    *,
    company: str,
    embedding_model: str,
    max_chunk_tokens: int,
    chunk_overlap_ratio: float,
) -> IndexManifest:
    if len(chunks) != vectors.shape[0]:
        raise ValueError(
            f"chunk/vector count mismatch: {len(chunks)} chunks, {vectors.shape[0]} vectors"
        )

    directory.mkdir(parents=True, exist_ok=True)

    with (directory / CHUNKS_FILE).open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(chunk.model_dump_json() + "\n")

    np.savez_compressed(directory / VECTORS_FILE, vectors=vectors.astype(np.float32))

    digest = hashlib.sha256((directory / CHUNKS_FILE).read_bytes()).hexdigest()

    manifest = IndexManifest(
        company=company,
        embedding_model=embedding_model,
        dimension=int(vectors.shape[1]) if vectors.size else 0,
        chunk_count=len(chunks),
        document_count=len({chunk.doc_id for chunk in chunks}),
        max_chunk_tokens=max_chunk_tokens,
        chunk_overlap_ratio=chunk_overlap_ratio,
        index_chunks_sha256=digest,
        built_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    (directory / MANIFEST_FILE).write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    logger.info("index written to %s (%d chunks)", directory, len(chunks))
    return manifest


def load_index(directory: Path) -> LoadedIndex:
    manifest_path = directory / MANIFEST_FILE
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"no index at {directory} — build it first with `python -m rag_ic.ingest.build_index`"
        )

    manifest = IndexManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if manifest.version != INDEX_VERSION:
        raise ValueError(
            f"index version {manifest.version} != expected {INDEX_VERSION}; rebuild the index"
        )

    chunks = [
        Chunk.model_validate_json(line)
        for line in (directory / CHUNKS_FILE).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with np.load(directory / VECTORS_FILE) as archive:
        vectors = archive["vectors"].astype(np.float32)

    if len(chunks) != vectors.shape[0]:
        raise ValueError("corrupt index: chunk count does not match vector rows")

    # Same count is not the same content. A re-ingest that changes the text while
    # preserving the number of chunks would otherwise pair every chunk with the
    # wrong vector and retrieve confident nonsense. Older indexes carry no digest,
    # so this only enforces where one was recorded.
    if manifest.index_chunks_sha256:
        actual = hashlib.sha256((directory / CHUNKS_FILE).read_bytes()).hexdigest()
        if actual != manifest.index_chunks_sha256:
            raise ValueError(
                "corrupt index: chunk file does not match the manifest digest; "
                "rebuild it with `python -m rag_ic.ingest.build_index`"
            )

    return LoadedIndex(chunks=chunks, vectors=vectors, manifest=manifest)

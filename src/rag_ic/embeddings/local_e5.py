"""Local multilingual-e5 embeddings via sentence-transformers.

The e5 family is trained with instruction prefixes and *silently* degrades without
them: a passage embedded without ``passage: `` still produces a vector, just a worse
one. Nothing errors, retrieval quality simply drops. That is why the prefixes live
in named functions with unit tests rather than inline string concatenation.

The model also truncates input above 512 tokens without warning, which is the
constraint the chunker is built around.
"""

from __future__ import annotations

import logging

import numpy as np

from rag_ic.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

#: Hard input limit of the e5 encoder. Text beyond this is dropped by the tokenizer.
MAX_MODEL_TOKENS = 512


def as_query(text: str) -> str:
    """Apply the e5 query prefix, without doubling it if already present."""
    stripped = text.strip()
    return stripped if stripped.startswith(QUERY_PREFIX) else f"{QUERY_PREFIX}{stripped}"


def as_passage(text: str) -> str:
    """Apply the e5 passage prefix, without doubling it if already present."""
    stripped = text.strip()
    return stripped if stripped.startswith(PASSAGE_PREFIX) else f"{PASSAGE_PREFIX}{stripped}"


def resolve_device(requested: str) -> str:
    """Map ``auto`` to cuda when a GPU is actually usable, else cpu."""
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover - torch is a hard dependency in practice
        return "cpu"


class E5EmbeddingProvider(EmbeddingProvider):
    """multilingual-e5 running locally on CPU or GPU."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        device: str = "auto",
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._device = resolve_device(device)
        self._batch_size = batch_size
        logger.info("loading embedding model %s on %s", model_name, self._device)
        self._model = SentenceTransformer(model_name, device=self._device)
        self._model.max_seq_length = MAX_MODEL_TOKENS

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._device

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(
            [as_passage(text) for text in texts],
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 256,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vector = self._model.encode(
            as_query(text),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vector, dtype=np.float32)

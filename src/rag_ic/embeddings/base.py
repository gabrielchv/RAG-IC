"""The embedding seam.

Queries and documents are embedded through *different* methods on purpose: e5-style
models are asymmetric and require different prefixes for each. Making that explicit
in the interface means a caller cannot silently get it wrong.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    """Turns text into L2-normalised float32 vectors."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier stored in the index manifest, so a stale index is detectable."""

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed corpus passages. Returns shape ``(len(texts), dimension)``."""

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single user query. Returns shape ``(dimension,)``."""

"""The generation seam.

This interface is the point of the whole project. The research claim is that a
company's documents can be indexed and searched entirely on its own hardware; only
generation currently leaves the machine. Making generation a one-method interface is
what turns "we could run this locally" from a promise in the paper into a config
change — see :mod:`rag_ic.llm.ollama`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class LLMProvider(ABC):
    """Streams a grounded answer given a system instruction and a user prompt."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier, e.g. ``gemini``. Recorded in results and eval output."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Concrete model identifier, e.g. ``gemini-2.5-flash``."""

    @property
    def is_local(self) -> bool:
        """Whether the retrieved context stays on this machine. Reported in the UI."""
        return False

    @abstractmethod
    def generate_stream(self, system_instruction: str, prompt: str) -> AsyncIterator[str]:
        """Yield answer text incrementally."""

    async def generate(self, system_instruction: str, prompt: str) -> str:
        """Collect the stream into a single string."""
        parts = [part async for part in self.generate_stream(system_instruction, prompt)]
        return "".join(parts)

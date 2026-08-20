"""Local generation via Ollama — the privacy end-state of this research.

STATUS: written, **not yet validated**. Ollama is not installed in the development
environment this was built in, so this class has never made a real request. Treat it
as the design being demonstrated rather than a measured result, and validate it
before citing any number produced through it.

Hardware note: the target machine has a GTX 1650 with 4 GB of VRAM. A 7B model
quantised to Q4_K_M needs roughly 4.4 GB and will spill into system RAM, which is
slow. A 3–4B model (``qwen3:4b``, ``llama3.2:3b``) fits and is the realistic choice
for the comparison against Gemini.

Setup::

    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull qwen3:4b
    # then in .env:
    RAG_LLM_PROVIDER=ollama
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx

from rag_ic.config import Settings
from rag_ic.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen3:4b",
        temperature: float = 0.1,
        timeout: float = 180.0,
    ) -> None:
        self._host = host.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._timeout = timeout

    @classmethod
    def from_settings(cls, settings: Settings) -> OllamaProvider:
        return cls(
            host=settings.ollama_host,
            model=settings.ollama_model,
            temperature=settings.gemini_temperature,
        )

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self._model

    @property
    def is_local(self) -> bool:
        return True

    async def generate_stream(self, system_instruction: str, prompt: str) -> AsyncIterator[str]:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "options": {"temperature": self._temperature},
        }
        async with (
            httpx.AsyncClient(timeout=self._timeout) as client,
            client.stream("POST", f"{self._host}/api/chat", json=payload) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                event = json.loads(line)
                content = event.get("message", {}).get("content")
                if content:
                    yield content
                if event.get("done"):
                    break

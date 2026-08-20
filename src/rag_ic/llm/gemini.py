"""Gemini generation via the google-genai SDK.

Cost notes for a proof of concept:

* ``thinking_level="low"`` keeps Gemini's internal reasoning tokens to a minimum.
  Those tokens are billed as output tokens and, for extractive question answering
  over five short chunks, they buy essentially nothing. Note that gemini-2.5-flash
  *cannot* disable thinking entirely — only reduce it.
* ``temperature=0.1`` — this is not a creative-writing task; near-greedy decoding
  makes the evaluation reproducible.

The thinking field has changed shape across SDK versions (an integer ``thinking_budget``
before, a categorical ``thinking_level`` now), so it is applied defensively: if the
installed SDK rejects it, the request still goes out with the model's default.

Model deprecation: ``gemini-2.5-flash`` is configuration, not code. If Google retires
it, change ``RAG_GEMINI_MODEL`` in ``.env``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from rag_ic.config import Settings
from rag_ic.llm.base import LLMProvider

logger = logging.getLogger(__name__)

#: Model families that accept a thinking config at all. Older ones reject the field.
_THINKING_CAPABLE_PREFIXES = ("gemini-2.5", "gemini-3")


class GeminiProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.1,
        max_output_tokens: int = 1500,
        thinking_level: str | None = "low",
    ) -> None:
        from google import genai

        if not api_key:
            raise ValueError(
                "no Gemini API key — set RAG_GEMINI_API_KEY in .env "
                "(get one at https://aistudio.google.com/apikey)"
            )
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._thinking_level = thinking_level

    @classmethod
    def from_settings(cls, settings: Settings) -> GeminiProvider:
        key = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else ""
        return cls(
            api_key=key,
            model=settings.gemini_model,
            temperature=settings.gemini_temperature,
            max_output_tokens=settings.gemini_max_output_tokens,
            thinking_level=settings.gemini_thinking_level,
        )

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def model(self) -> str:
        return self._model

    def _config(self, system_instruction: str):
        from google.genai import types

        kwargs = {
            "system_instruction": system_instruction,
            "temperature": self._temperature,
            "max_output_tokens": self._max_output_tokens,
        }
        thinking = self._thinking_config(types)
        if thinking is not None:
            kwargs["thinking_config"] = thinking
        return types.GenerateContentConfig(**kwargs)

    def _thinking_config(self, types):
        """Best-effort thinking config, tolerant of SDK version drift.

        Returns ``None`` when the model does not support thinking, when the feature is
        switched off in settings, or when the installed SDK does not know the field —
        in which case the request simply uses the model's default reasoning effort.
        """
        if self._thinking_level is None or not self._model.startswith(_THINKING_CAPABLE_PREFIXES):
            return None
        try:
            return types.ThinkingConfig(thinking_level=self._thinking_level)
        except Exception as exc:
            logger.warning(
                "installed google-genai does not accept thinking_level=%r (%s); "
                "using the model default reasoning effort",
                self._thinking_level,
                exc,
            )
            return None

    async def generate_stream(self, system_instruction: str, prompt: str) -> AsyncIterator[str]:
        stream = await self._client.aio.models.generate_content_stream(
            model=self._model,
            contents=prompt,
            config=self._config(system_instruction),
        )
        async for event in stream:
            text = getattr(event, "text", None)
            if text:
                yield text

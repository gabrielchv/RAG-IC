"""Gemini generation via the google-genai SDK.

Cost notes for a proof of concept:

* Reasoning effort is kept low. Thinking tokens are billed as output tokens and, for
  extractive question answering over five short chunks, they buy essentially nothing.
* ``temperature=0.1`` — this is not a creative-writing task; near-greedy decoding
  makes the evaluation reproducible.

The thinking knob is *not* the same field across model families, and getting it wrong is
a request-time 400 rather than a type error:

* Gemini 3 takes the categorical ``thinking_level`` ("minimal"/"low"/"medium"/"high").
* Gemini 2.5 takes the integer ``thinking_budget``, and rejects ``thinking_level``
  outright — ``400 INVALID_ARGUMENT ... Thinking level is not supported for this
  model.`` On 2.5 Flash a budget of 0 does disable thinking entirely.

So settings expose one semantic level and this module maps it to whichever field the
configured model actually accepts. As a backstop for model families that are neither,
a 400 that mentions thinking disables the field for the rest of the process and the
request is retried once with the model's default reasoning effort.

Model deprecation: ``gemini-2.5-flash`` is configuration, not code. If Google retires
it, change ``RAG_GEMINI_MODEL`` in ``.env``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from rag_ic.config import Settings
from rag_ic.llm.base import LLMProvider

logger = logging.getLogger(__name__)

#: Gemini 2.5 spends thinking by token budget. The mapping is deliberately frugal:
#: this is extractive QA over five short chunks, not a reasoning benchmark.
_THINKING_BUDGETS = {"minimal": 0, "low": 512, "medium": 4096, "high": 12288}


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
        #: Set when the API rejects the thinking field, so it is tried only once.
        self._thinking_rejected = False

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
        """Map the configured reasoning level onto the field this model accepts.

        Returns ``None`` — meaning "use the model default" — when thinking is switched
        off in settings, when the model family takes neither field, or once the API has
        rejected the field for this process.
        """
        if self._thinking_level is None or self._thinking_rejected:
            return None
        if self._model.startswith("gemini-3"):
            return types.ThinkingConfig(thinking_level=self._thinking_level)
        if self._model.startswith("gemini-2.5"):
            budget = _THINKING_BUDGETS.get(self._thinking_level)
            return None if budget is None else types.ThinkingConfig(thinking_budget=budget)
        return None

    def _note_thinking_rejected(self, exc: Exception) -> bool:
        """Was this failure the thinking field? If so, stop sending it and say so."""
        if self._thinking_rejected or self._thinking_level is None:
            return False
        if "thinking" not in str(exc).lower():
            return False
        self._thinking_rejected = True
        logger.warning(
            "%s rejected the thinking field (%s); retrying with the model default reasoning effort",
            self._model,
            exc,
        )
        return True

    async def generate_stream(self, system_instruction: str, prompt: str) -> AsyncIterator[str]:
        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=self._model,
                contents=prompt,
                config=self._config(system_instruction),
            )
        except Exception as exc:
            if not self._note_thinking_rejected(exc):
                raise
            # _config now omits the thinking field; nothing has been yielded yet.
            stream = await self._client.aio.models.generate_content_stream(
                model=self._model,
                contents=prompt,
                config=self._config(system_instruction),
            )
        async for event in stream:
            text = getattr(event, "text", None)
            if text:
                yield text

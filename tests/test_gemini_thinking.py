"""The thinking field is not the same field across Gemini families.

Sending ``thinking_level`` to a 2.5 model is a request-time ``400 INVALID_ARGUMENT``,
not a type error, so nothing catches it before the tokens are already spent on a failed
call. These tests pin the mapping without touching the network.
"""

from __future__ import annotations

from google.genai import types

from rag_ic.llm.gemini import GeminiProvider


def make_provider(model: str, level: str | None = "low") -> GeminiProvider:
    return GeminiProvider(api_key="test-key-not-used", model=model, thinking_level=level)


def test_gemini_25_gets_a_token_budget_never_a_level() -> None:
    config = make_provider("gemini-2.5-flash")._thinking_config(types)
    assert config is not None
    assert config.thinking_level is None
    assert config.thinking_budget == 512


def test_minimal_disables_thinking_on_gemini_25() -> None:
    config = make_provider("gemini-2.5-flash", "minimal")._thinking_config(types)
    assert config is not None
    assert config.thinking_budget == 0


def test_gemini_3_gets_the_categorical_level() -> None:
    config = make_provider("gemini-3-flash")._thinking_config(types)
    assert config is not None
    # The SDK coerces the string onto its own enum.
    assert config.thinking_level.value.lower() == "low"
    assert config.thinking_budget is None


def test_no_thinking_config_when_switched_off() -> None:
    assert make_provider("gemini-2.5-flash", None)._thinking_config(types) is None


def test_unknown_model_family_uses_the_model_default() -> None:
    assert make_provider("some-future-model")._thinking_config(types) is None


def test_a_rejection_stops_the_field_being_sent_again() -> None:
    provider = make_provider("gemini-2.5-flash")
    error = Exception("400 INVALID_ARGUMENT. Thinking level is not supported for this model.")

    assert provider._note_thinking_rejected(error) is True
    assert provider._thinking_config(types) is None
    # Only once: a second failure is a different problem and must surface.
    assert provider._note_thinking_rejected(error) is False


def test_an_unrelated_failure_is_not_swallowed() -> None:
    provider = make_provider("gemini-2.5-flash")
    assert provider._note_thinking_rejected(Exception("503 UNAVAILABLE")) is False
    assert provider._thinking_config(types) is not None

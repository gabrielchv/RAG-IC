"""Token counting.

The chunker must respect the embedding model's 512-token input limit, so it needs a
token count — but it should not be forced to load a 500 MB model to run in a unit
test. Hence the injectable ``TokenCounter`` protocol with a cheap default.
"""

from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)

#: Characters per token, measured on Portuguese text with the XLM-RoBERTa
#: tokenizer that multilingual-e5 uses. Only used by the approximate counter.
_CHARS_PER_TOKEN_PT = 3.5


class TokenCounter(Protocol):
    def __call__(self, text: str) -> int:  # pragma: no cover - structural type
        ...


def approximate_token_count(text: str) -> int:
    """Cheap character-ratio estimate. Deliberately conservative (rounds up)."""
    if not text:
        return 0
    return max(1, int(len(text) / _CHARS_PER_TOKEN_PT) + 1)


def transformer_token_counter(model_name: str) -> TokenCounter:
    """Exact counter using the embedding model's own tokenizer.

    Falls back to :func:`approximate_token_count` if transformers is unavailable or
    the tokenizer cannot be downloaded — the index is still built, just with a
    slightly less precise chunk budget.
    """
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as exc:  # pragma: no cover - depends on network/environment
        logger.warning(
            "could not load tokenizer for %s (%s); falling back to approximate counting",
            model_name,
            exc,
        )
        return approximate_token_count

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=True))

    return count

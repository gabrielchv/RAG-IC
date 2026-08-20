"""BM25 lexical retrieval.

Dense embeddings are good at paraphrase and bad at identifiers. In this domain the
identifiers *are* the question: "qual o CA da luva de vaqueta?", "o que diz o item
6.6.1?", "treinamento de NR-35". A 384-dimensional vector blurs ``CA 31.028`` and
``CA 31.029`` into near-identical points; BM25 does not.

So the tokenizer here is the important part, not the ranking function:

* diacritics are folded, because users type "protecao" as often as "proteção";
* compound identifiers survive as single tokens — ``nr-35``, ``6.6.1``, ``31.028``;
* those compounds *also* emit their parts, so "NR 35" matches "NR-35".
"""

from __future__ import annotations

import re
import unicodedata

from rank_bm25 import BM25Okapi

from rag_ic.models import Chunk

#: A token is a run of letters/digits, optionally joined by . - / into one identifier.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[./-][a-z0-9]+)*")
_SPLIT_RE = re.compile(r"[./-]")

#: Deliberately short: aggressive stopword removal hurts phrase-like queries, and
#: BM25's IDF term already discounts ubiquitous words.
STOPWORDS_PT = frozenset(
    """
    a ao aos as as com como da das de do dos e em entre para pela pelas pelo pelos
    por que se sem sob sobre um uma umas uns o os na nas no nos ou e/ou ser sao e
    """.split()  # noqa: SIM905 - readable block form beats a 40-item literal
)

#: Sub-tokens shorter than this are dropped: splitting "6.3.1" into "6", "3", "1"
#: adds noise without adding signal.
_MIN_SUBTOKEN_LENGTH = 2


def fold_diacritics(text: str) -> str:
    """ "Proteção" -> "protecao". Applied identically to corpus and query."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def tokenize(text: str) -> list[str]:
    """Lowercase, fold diacritics, keep identifiers whole and also emit their parts."""
    normalised = fold_diacritics(text.lower())
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(normalised):
        token = match.group(0)
        if token in STOPWORDS_PT:
            continue
        tokens.append(token)
        if _SPLIT_RE.search(token):
            tokens.extend(
                part
                for part in _SPLIT_RE.split(token)
                if len(part) >= _MIN_SUBTOKEN_LENGTH and part not in STOPWORDS_PT
            )
    return tokens


class BM25Index:
    """BM25 over the chunks' indexable text, rebuilt at load time.

    Rebuilding costs a few hundred milliseconds for a corpus this size, which is far
    cheaper than owning a serialisation format for it.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        self._corpus_tokens = [tokenize(chunk.indexable_text) for chunk in chunks]
        # BM25Okapi rejects an empty corpus; guard so an empty index still loads.
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Return ``(chunk_index, score)`` for the ``top_k`` best matches, score-sorted."""
        if self._bm25 is None or top_k <= 0:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
        return [(index, float(score)) for index, score in ranked[:top_k] if score > 0.0]

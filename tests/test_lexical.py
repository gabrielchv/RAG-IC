"""The BM25 tokenizer is what makes identifier queries work. It is the whole point."""

from __future__ import annotations

from rag_ic.models import Chunk
from rag_ic.retrieval.lexical import BM25Index, fold_diacritics, tokenize


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc",
        doc_title="Doc",
        source_path="doc.md",
        source_format="markdown",
        text=text,
        token_count=len(text.split()),
        ordinal=0,
    )


def test_fold_diacritics():
    assert fold_diacritics("Proteção Auditiva") == "Protecao Auditiva"


def test_query_without_accents_matches_corpus_with_accents():
    assert tokenize("protecao") == tokenize("proteção")


def test_identifiers_survive_as_single_tokens():
    tokens = tokenize("Conforme a NR-35 e o item 6.6.1 do CA 31.028")
    assert "nr-35" in tokens
    assert "6.6.1" in tokens
    assert "31.028" in tokens


def test_identifier_parts_are_also_emitted():
    # "NR 35" written with a space in the query must still reach a document
    # that spells it "NR-35".
    tokens = tokenize("NR-35")
    assert "nr" in tokens
    assert "35" in tokens


def test_short_subtokens_are_dropped():
    # Splitting 6.6.1 into 6, 6, 1 would add noise with no discriminative power.
    tokens = tokenize("item 6.6.1")
    assert "6.6.1" in tokens
    assert "6" not in tokens
    assert "1" not in tokens


def test_stopwords_are_removed():
    assert tokenize("a luva de raspa") == ["luva", "raspa"]


def test_bm25_ranks_the_exact_identifier_first():
    index = BM25Index(
        [
            make_chunk("a", "A luva de vaqueta protege contra corte por cantos vivos."),
            make_chunk("b", "A luva de raspa cano 20 cm tem CA 31.028 e dura 2 meses."),
            make_chunk("c", "O capacete classe B tem CA 31.469 e dura 5 anos."),
        ]
    )
    hits = index.search("CA 31.028", top_k=3)
    assert hits, "BM25 returned nothing for an exact identifier"
    assert hits[0][0] == 1


def test_bm25_on_empty_corpus_returns_nothing():
    assert BM25Index([]).search("qualquer coisa", top_k=5) == []


def test_bm25_with_only_stopwords_returns_nothing():
    index = BM25Index([make_chunk("a", "conteúdo qualquer")])
    assert index.search("a de o", top_k=5) == []

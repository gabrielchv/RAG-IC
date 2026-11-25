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


def test_matches_are_returned_when_every_chunk_contains_the_query_term():
    # BM25Okapi's IDF is negative once a term appears in more than half the
    # documents, so every score here is negative. Filtering on `score > 0.0`
    # returned [] for a query that matches the whole corpus — and since
    # indexable_text prefixes the document header to every chunk, any query
    # naming the document hits this.
    index = BM25Index([make_chunk(f"c{i}", "Política de EPI. Fornecimento de luvas.") for i in range(10)])
    hits = index.search("Política de EPI", top_k=5)

    assert hits, "BM25 dropped every chunk even though all of them match"
    assert len(hits) == 5


def test_non_matching_chunks_are_still_excluded():
    index = BM25Index(
        [
            make_chunk("a", "Política de EPI. Fornecimento de luvas."),
            make_chunk("b", "Política de EPI. Fornecimento de botas."),
            make_chunk("c", "Assunto completamente diferente sobre ruído."),
        ]
    )
    hits = index.search("luvas", top_k=5)

    assert [i for i, _ in hits] == [0], hits


def test_results_are_ordered_by_score_descending():
    index = BM25Index(
        [
            make_chunk("a", "luva luva luva luva capacete"),
            make_chunk("b", "luva capacete capacete capacete"),
            make_chunk("c", "totalmente outro assunto"),
        ]
    )
    hits = index.search("luva", top_k=3)

    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True), scores

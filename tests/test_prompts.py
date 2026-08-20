"""The prompt contract: numbered sources, machine-checkable citations, explicit refusal."""

from __future__ import annotations

from rag_ic.models import INSUFFICIENT_CONTEXT_MARKER, Chunk, ScoredChunk
from rag_ic.prompts import (
    build_system_prompt,
    build_user_prompt,
    format_sources,
    is_refusal,
    strip_refusal_marker,
)


def source(title: str, section: tuple[str, ...], text: str) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=f"{title}#0",
            doc_id=title,
            doc_title=title,
            source_path=f"{title}.md",
            source_format="markdown",
            section_path=section,
            text=text,
            token_count=10,
            ordinal=0,
        )
    )


def test_system_prompt_names_the_company_and_the_marker():
    prompt = build_system_prompt("Metalúrgica Aurora Ltda.")
    assert "Metalúrgica Aurora Ltda." in prompt
    assert INSUFFICIENT_CONTEXT_MARKER in prompt


def test_sources_are_numbered_from_one():
    rendered = format_sources(
        [source("Doc A", ("1. Um",), "texto A"), source("Doc B", ("2. Dois",), "texto B")]
    )
    assert rendered.startswith("[1] Doc A § 1. Um")
    assert "[2] Doc B § 2. Dois" in rendered


def test_user_prompt_contains_the_question_and_the_sources():
    prompt = build_user_prompt("Qual o CA da luva?", [source("Doc", ("1. Um",), "CA 27.033")])
    assert "PERGUNTA: Qual o CA da luva?" in prompt
    assert "CA 27.033" in prompt


def test_user_prompt_survives_an_empty_retrieval():
    prompt = build_user_prompt("Qualquer coisa?", [])
    assert "nenhum trecho relevante" in prompt
    assert "PERGUNTA: Qualquer coisa?" in prompt


def test_refusal_detection_requires_the_marker_at_the_start():
    assert is_refusal(f"{INSUFFICIENT_CONTEXT_MARKER}\nFalta a ficha técnica.")
    assert is_refusal(f"  {INSUFFICIENT_CONTEXT_MARKER}: falta o documento.")
    # A mention in the middle of a real answer must not count as a refusal.
    assert not is_refusal(f"A troca é semestral [1]. Não é caso de {INSUFFICIENT_CONTEXT_MARKER}.")


def test_strip_refusal_marker_leaves_the_explanation():
    stripped = strip_refusal_marker(f"{INSUFFICIENT_CONTEXT_MARKER}: procure a Coordenação de SST.")
    assert stripped == "procure a Coordenação de SST."


def test_strip_refusal_marker_is_a_noop_on_normal_answers():
    answer = "A troca é semestral [1]."
    assert strip_refusal_marker(answer) == answer

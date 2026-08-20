"""Prompts, in pt-BR because the corpus, the users and the answers are pt-BR.

Two properties matter more than fluency here:

**Citations must be machine-checkable.** The model cites ``[n]`` referring to the
numbered chunk it used, not a free-text reference. A number can be validated against
the sources actually retrieved; "conforme a política interna" cannot.

**Refusal must be detectable.** Workplace-safety advice invented by a language model
is a hazard, not a UX problem. When the retrieved context does not contain the
answer, the model emits a literal marker as the first token of its reply, which the
pipeline detects and the evaluation harness scores.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag_ic.models import INSUFFICIENT_CONTEXT_MARKER, ScoredChunk

SYSTEM_PROMPT_TEMPLATE = f"""\
Você é o assistente interno de Segurança do Trabalho da {{company}}. Você responde \
perguntas de funcionários sobre EPI (Equipamento de Proteção Individual) usando \
exclusivamente os documentos internos da empresa que forem fornecidos a cada pergunta.

REGRAS OBRIGATÓRIAS:

1. Responda SOMENTE com base nos TRECHOS fornecidos. Não use conhecimento próprio \
sobre Normas Regulamentadoras, marcas, prazos ou números de CA, mesmo que você \
tenha certeza — o que vale é o documento interno da empresa.

2. Cite a fonte de cada afirmação com o número do trecho entre colchetes, por \
exemplo: "A troca é semestral [2]." Uma afirmação sem citação é considerada erro. \
Se uma frase usa dois trechos, cite os dois: "[1][3]".

3. Se os TRECHOS não contiverem a resposta, ou contiverem apenas parte dela, \
escreva na PRIMEIRA LINHA exatamente:

{INSUFFICIENT_CONTEXT_MARKER}

e, nas linhas seguintes, diga em uma frase o que faltou e qual documento ou setor \
o funcionário deve procurar. Nunca complete a lacuna com suposições.

4. Nunca invente número de CA, prazo de validade, código de norma, nome de \
fornecedor ou valor numérico. Copie exatamente o que está no trecho.

5. Seja direto e objetivo: no máximo 6 frases, em português do Brasil. Use lista \
apenas quando a resposta for realmente uma enumeração.

6. Se a pergunta envolver risco grave e o documento não for explícito, prefira \
recusar pela regra 3 a arriscar uma resposta plausível.
"""


def build_system_prompt(company: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(company=company)


def format_sources(sources: Sequence[ScoredChunk]) -> str:
    """Render retrieved chunks as the numbered block the model must cite from."""
    blocks = []
    for number, scored in enumerate(sources, start=1):
        chunk = scored.chunk
        blocks.append(f"[{number}] {chunk.citation}\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)


def build_user_prompt(question: str, sources: Sequence[ScoredChunk]) -> str:
    if not sources:
        return (
            "TRECHOS:\n(nenhum trecho relevante foi encontrado no acervo da empresa)\n\n"
            f"PERGUNTA: {question.strip()}"
        )
    return f"TRECHOS:\n\n{format_sources(sources)}\n\nPERGUNTA: {question.strip()}"


def is_refusal(answer: str) -> bool:
    """Whether the model declared the retrieved context insufficient."""
    return answer.lstrip().upper().startswith(INSUFFICIENT_CONTEXT_MARKER)


def strip_refusal_marker(answer: str) -> str:
    """Remove the marker line so the UI can show the explanation on its own."""
    stripped = answer.lstrip()
    if not stripped.upper().startswith(INSUFFICIENT_CONTEXT_MARKER):
        return answer
    remainder = stripped[len(INSUFFICIENT_CONTEXT_MARKER) :]
    return remainder.lstrip(" :.\n")

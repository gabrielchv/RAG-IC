"""Loaders: structure in, blocks out."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_ic.ingest.loaders import doc_id_for, load_corpus, load_csv, load_document, load_markdown

MARKDOWN = """\
# Política de EPI

Parágrafo de abertura.

## 1. Objetivo

Estabelecer regras.

### 1.1 Escopo

Vale para todos.

## 2. Penalidades

Advertência verbal.
"""


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    (tmp_path / "politica.md").write_text(MARKDOWN, encoding="utf-8")
    (tmp_path / "matriz.csv").write_text(
        "funcao,epi,ca\nSoldador,Luva de raspa,31028\nPintor,Luva nitrílica,36741\n",
        encoding="utf-8",
    )
    (tmp_path / "ignorar.xlsx").write_bytes(b"binario")
    return tmp_path


def test_doc_id_is_the_relative_path_without_suffix(tmp_path: Path):
    path = tmp_path / "pdf" / "comunicado.pdf"
    assert doc_id_for(path, tmp_path) == "pdf/comunicado"


def test_markdown_title_comes_from_the_h1(corpus: Path):
    document = load_markdown(corpus / "politica.md", "politica")
    assert document.title == "Política de EPI"


def test_h1_is_excluded_from_section_paths(corpus: Path):
    document = load_markdown(corpus / "politica.md", "politica")
    opening = next(block for block in document.blocks if "abertura" in block.text)
    assert opening.section_path == ()


def test_nested_headings_build_the_section_path(corpus: Path):
    document = load_markdown(corpus / "politica.md", "politica")
    nested = next(block for block in document.blocks if "todos" in block.text)
    assert nested.section_path == ("1. Objetivo", "1.1 Escopo")


def test_a_deeper_heading_pops_back_at_the_same_level(corpus: Path):
    document = load_markdown(corpus / "politica.md", "politica")
    penalties = next(block for block in document.blocks if "Advertência" in block.text)
    assert penalties.section_path == ("2. Penalidades",)


def test_csv_produces_one_atomic_block_per_row(corpus: Path):
    document = load_csv(corpus / "matriz.csv", "matriz")
    assert len(document.blocks) == 2
    assert all(block.atomic for block in document.blocks)
    assert document.blocks[0].row == 1
    assert "funcao: Soldador" in document.blocks[0].text
    assert "ca: 31028" in document.blocks[0].text


def test_csv_rows_are_rendered_as_column_value_pairs(corpus: Path):
    document = load_csv(corpus / "matriz.csv", "matriz")
    assert document.blocks[1].text == "funcao: Pintor; epi: Luva nitrílica; ca: 36741"


def test_load_corpus_skips_unsupported_files_and_sorts(corpus: Path):
    documents = load_corpus(corpus)
    assert [document.doc_id for document in documents] == ["matriz", "politica"]


def test_unsupported_suffix_raises(tmp_path: Path):
    path = tmp_path / "planilha.xlsx"
    path.write_bytes(b"x")
    with pytest.raises(ValueError, match="unsupported"):
        load_document(path, tmp_path)


def test_missing_corpus_directory_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "nao-existe")

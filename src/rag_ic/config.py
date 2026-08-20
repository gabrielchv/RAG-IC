"""Application configuration, loaded from environment / .env."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

RetrievalStrategy = Literal["dense", "bm25", "hybrid"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings.

    Every field can be overridden with a ``RAG_``-prefixed environment variable.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RAG_",
        extra="ignore",
    )

    # --- Tenancy -----------------------------------------------------------
    company: str = "aurora"
    company_display_name: str = "Metalúrgica Aurora Ltda."
    data_dir: Path = PROJECT_ROOT / "data"
    index_dir: Path = PROJECT_ROOT / "index"
    web_dir: Path = PROJECT_ROOT / "web"

    # --- Embeddings --------------------------------------------------------
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_device: str = "auto"
    embedding_batch_size: int = 32

    # --- Chunking ----------------------------------------------------------
    max_chunk_tokens: int = 400
    chunk_overlap_ratio: float = Field(default=0.15, ge=0.0, lt=0.5)

    # --- Retrieval ---------------------------------------------------------
    retrieval_strategy: RetrievalStrategy = "hybrid"
    candidates_per_retriever: int = 30
    top_k: int = 5
    rrf_k: int = 60

    # --- Generation --------------------------------------------------------
    llm_provider: Literal["gemini", "ollama"] = "gemini"
    gemini_model: str = "gemini-2.5-flash"
    gemini_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("RAG_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"),
    )
    gemini_temperature: float = 0.1
    gemini_max_output_tokens: int = 1500
    #: Reasoning effort. Thinking tokens are billed as output tokens, so "low" is the
    #: cheapest setting available — gemini-2.5-flash cannot disable thinking entirely.
    #: Set to None to leave the model default alone.
    gemini_thinking_level: Literal["minimal", "low", "medium", "high"] | None = "low"

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"

    # --- Misc --------------------------------------------------------------
    log_level: str = "INFO"

    @property
    def company_docs_dir(self) -> Path:
        """Source documents for the active company."""
        return self.data_dir / self.company / "docs"

    @property
    def company_gold_dir(self) -> Path:
        """Gold evaluation set for the active company."""
        return self.data_dir / self.company / "gold"

    @property
    def company_index_dir(self) -> Path:
        """Built index for the active company."""
        return self.index_dir / self.company


def get_settings() -> Settings:
    """Build settings from the environment.

    Not cached: tests and CLI entry points override the environment freely.
    """
    return Settings()

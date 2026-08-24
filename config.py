# ============================================================
# config.py
# Centralized, typed configuration for the RAG pipeline and
# benchmark runner. Supports multiple LLM providers and multiple
# benchmark datasets.
# ============================================================

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for the LLM used by the RAG pipeline."""

    provider: Literal["ollama", "openai", "anthropic", "google"]
    model: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    is_thinking_model: bool = False
    thinking_config: dict[str, Any] | None = None
    auto_pull: bool = True  # Only used when provider == "ollama"


class ChunkingConfig(BaseModel):
    """Configuration for document chunking."""

    chunk_size: int = 500
    chunk_overlap: int = 50


class EmbeddingConfig(BaseModel):
    """Configuration for the embedding model."""

    model_name: str = "BAAI/bge-small-en-v1.5"


class RetrievalConfig(BaseModel):
    """Configuration for the vector-store retriever."""

    k: int = 5


class SquadV2Config(BaseModel):
    """Configuration for the SQuAD v2 benchmark."""

    dataset_name: str = "rajpurkar/squad_v2"
    split: str = "validation"
    num_questions: int = 30
    seed: int = 42


class BenchmarksConfig(BaseModel):
    """Configuration for all available benchmarks.

    The `enabled` list controls which benchmarks are run when
    `benchmark_eval.py` is executed. Each benchmark has its own
    dedicated config key below.

    Verbosity levels:
        0: progress bar only (default)
        1: print each question/answer with "Q x/total" progress
    """

    enabled: list[str] = Field(default_factory=lambda: ["squad_v2"])
    verbosity: int = 0
    squad_v2: SquadV2Config = Field(default_factory=SquadV2Config)


class Config(BaseModel):
    """Root configuration object."""

    model: ModelConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    benchmarks: BenchmarksConfig


# ------------------------------------------------------------------
# Default configuration. Edit this to switch models, providers, or
# benchmark parameters.
# ------------------------------------------------------------------
CONFIG = Config(
    model=ModelConfig(
        provider="ollama",
        model="llama3.2:1b",
        # model="qwen3:30b",
        parameters={"temperature": 0},
        is_thinking_model=False,
        auto_pull=True,
    ),
    chunking=ChunkingConfig(chunk_size=500, chunk_overlap=50),
    embedding=EmbeddingConfig(model_name="BAAI/bge-large-en-v1.5"),
    # embedding=EmbeddingConfig(model_name="BAAI/bge-small-en-v1.5"),
    retrieval=RetrievalConfig(k=5),
    benchmarks=BenchmarksConfig(
        enabled=["squad_v2"],
        squad_v2=SquadV2Config(num_questions=100, seed=7),
    ),
)

# ============================================================
# config.py
# Centralized, typed configuration for the RAG pipeline and
# benchmark runner. Supports multiple LLM providers and multiple
# benchmark datasets.
# ============================================================

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


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


class DocumentsConfig(BaseModel):
    """Configuration for the documents the pipeline answers questions about."""

    source: str = "Documents"  # A single PDF file or a folder of PDFs
    persist_root: str = "chroma_db"  # Each source gets its own store under here
    rebuild: bool = False  # Discard the cached index for `source` and re-embed


class SquadV2Config(BaseModel):
    """Configuration for the SQuAD v2 benchmark."""

    dataset_name: str = "rajpurkar/squad_v2"
    split: str = "validation"
    num_questions: int = 30
    seed: int = 42


class MmluChemConfig(BaseModel):
    """Configuration for the MMLU-Chem multiple-choice benchmark.

    With ``use_rag=False`` (default) the model answers from its own knowledge.
    With ``use_rag=True`` each question is answered with context retrieved
    from the documents in ``knowledge_source``.
    """

    dataset_name: str = "cais/mmlu"
    subject: str = "college_chemistry"
    split: str = "test"
    num_questions: int = 20
    seed: int = 42
    use_rag: bool = False
    knowledge_source: str | None = None  # Folder of PDFs forming the knowledge base
    rebuild_index: bool = False  # Re-embed knowledge_source instead of reusing its cache

    @model_validator(mode="after")
    def _require_knowledge_source(self) -> MmluChemConfig:
        if self.use_rag and not self.knowledge_source:
            raise ValueError("mmlu_chem.use_rag=True requires knowledge_source to be set")
        return self


class BenchmarksConfig(BaseModel):
    """Configuration for all available benchmarks.

    The `enabled` list controls which benchmarks are run when
    `benchmark_eval.py` is executed. Each benchmark has its own
    dedicated config key below.
    Options: squad_v2, mmlu_chem
    Verbosity levels:
        0: progress bar only (default)
        1: print each question/answer with "Q x/total" progress
    """

    enabled: list[str] = Field(default_factory=lambda: ["squad_v2", "mmlu_chem"])
    verbosity: int = 0
    squad_v2: SquadV2Config = Field(default_factory=SquadV2Config)
    mmlu_chem: MmluChemConfig = Field(default_factory=MmluChemConfig)


class Config(BaseModel):
    """Root configuration object."""

    model: ModelConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    benchmarks: BenchmarksConfig
    documents: DocumentsConfig = Field(default_factory=DocumentsConfig)


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
        enabled=["squad_v2", "mmlu_chem"],
        squad_v2=SquadV2Config(num_questions=100, seed=7),
        mmlu_chem=MmluChemConfig(
            num_questions=100,
            seed=42,
            use_rag=False,
            # use_rag=True,
            # knowledge_source="Documents/chemistry",
        ),
    ),
    documents=DocumentsConfig(
        source="Documents",
        # source="Documents/mesh_splatting.pdf",
        rebuild=False,
    ),
)

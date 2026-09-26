# Local RAG Pipeline

A configurable, local-first Retrieval-Augmented Generation (RAG) pipeline. Ask questions grounded in your PDF documents using open-source or commercial LLMs.

## Overview

This project builds a RAG pipeline that:

- Loads a single PDF or every PDF in a folder.
- Splits them into overlapping chunks.
- Generates embeddings and stores vectors in ChromaDB.
- Retrieves relevant context for a question.
- Uses an LLM to generate an answer based only on the retrieved context.

It also includes a benchmark runner that evaluates the pipeline on SQuAD v2 and MMLU-Chem, measuring answer accuracy, hallucination rate, and generation performance (tokens/second).

## Features

- **PDF document ingestion** — load all PDFs from a directory.
- **Chunking & embeddings** — configurable chunk size, overlap, and embedding model.
- **Vector storage** — ChromaDB for fast similarity search.
- **Multiple LLM providers** — Ollama (default), OpenAI, Anthropic, Google Gemini.
- **Thinking-model support** — captures reasoning/thinking content separately from the final answer.
- **Ollama auto-pull** — missing Ollama models are pulled automatically with a progress bar.
- **Config-driven** — all settings live in `config.py`.
- **Benchmarking** — evaluate on SQuAD v2 (exact-match, contains-answer, hallucination) and MMLU-Chem (multiple-choice accuracy, with or without RAG).
- **Performance metrics** — per-question latency, output tokens, and tokens/second.
- **Verbosity levels** — quiet progress bar or per-question Q/A output.
- **Organized results** — benchmark outputs saved under `results/<benchmark>/<model>/<timestamp>/` as CSV + JSON summary.

## Tech stack

- Python 3.12
- LangChain + LangChain Community
- ChromaDB
- Hugging Face Sentence Transformers
- Ollama
- Optional: OpenAI, Anthropic, Google Gemini APIs

## Project structure

```text
.
├── config.py              # Centralized configuration (model, chunking, retrieval, benchmarks)
├── model_utils.py         # LLM factory, Ollama auto-pull, thinking-content extraction
├── main.py                # Run the RAG pipeline on your documents
├── benchmark_eval.py      # Run SQuAD v2 / MMLU-Chem benchmarks
├── environment.yml        # Mamba/Conda environment definition
├── requirements.txt       # Pip requirements (works with uv pip)
├── Documents/             # Your PDFs (committed)
├── chroma_db/             # One vector store per source (created on first run)
└── results/               # Benchmark outputs (created on benchmark runs)
```

## Installation

### Option A: Mamba / Conda

```bash
mamba env create -f environment.yml
mamba activate rag_pipeline
```

> The `environment.yml` uses `conda-forge` and Python 3.12. It works with both Mamba and Conda.

### Option B: uv

```bash
uv venv .venv --python 3.12
uv pip install -r requirements.txt
source .venv/bin/activate
```

### Option C: pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

All tunable settings are in `config.py`. Edit the `CONFIG = Config(...)` block at the bottom of the file: values set there override the defaults in the config classes above it.

### Default model (Ollama)

```python
CONFIG = Config(
    model=ModelConfig(
        provider="ollama",
        model="llama3.2:1b",
        parameters={"temperature": 0},
        is_thinking_model=False,
        auto_pull=True,
    ),
    ...
)
```

If `auto_pull=True` and the Ollama model is not available locally, it will be downloaded automatically before running.

### Switch to OpenAI

```python
model=ModelConfig(
    provider="openai",
    model="gpt-4o-mini",
    parameters={"temperature": 0},
)
```

Set your API key:

```bash
export OPENAI_API_KEY=your-key
```

### Switch to Anthropic

```python
model=ModelConfig(
    provider="anthropic",
    model="claude-3-5-sonnet-latest",
    parameters={"temperature": 0},
)
```

Set your API key:

```bash
export ANTHROPIC_API_KEY=your-key
```

### Switch to Google Gemini

```python
model=ModelConfig(
    provider="google",
    model="gemini-2.0-flash",
    parameters={"temperature": 0},
)
```

Set your API key:

```bash
export GOOGLE_API_KEY=your-key
```

### Thinking models

For models that emit reasoning (e.g., DeepSeek-R1, QwQ, o1/o3, Gemini 2.5 Flash Thinking):

```python
model=ModelConfig(
    provider="ollama",
    model="deepseek-r1:8b",
    parameters={"temperature": 0.6},
    is_thinking_model=True,
)
```

The benchmark runner will save both the reasoning content and the final answer in the results CSV and JSON summary.

### Documents

```python
documents=DocumentsConfig(
    source="Documents",                   # a folder of PDFs...
    # source="Documents/mesh_splatting.pdf",  # ...or a single PDF
    rebuild=False,                        # True = discard the cached index and re-embed
),
```

Each source gets its own index under `chroma_db/<name>-<hash>`, where the hash covers the source's absolute path and the embedding model. Switching documents or embedding models therefore never reuses a stale index. Set `rebuild=True` after editing a PDF in place.

### Benchmark settings

```python
benchmarks=BenchmarksConfig(
    enabled=["squad_v2", "mmlu_chem"],
    verbosity=0,  # 0 = progress bar, 1 = show each Q/A
    squad_v2=SquadV2Config(
        dataset_name="rajpurkar/squad_v2",
        split="validation",
        num_questions=30,
        seed=42,
    ),
    mmlu_chem=MmluChemConfig(
        subject="college_chemistry",
        num_questions=20,
        seed=42,
        use_rag=False,  # True = answer with context retrieved from knowledge_source
        knowledge_source=None,  # e.g. "Documents/chemistry" (required when use_rag=True)
    ),
)
```

`enabled` lists the benchmarks to run (options: `squad_v2`, `mmlu_chem`); remove one to skip it.

**MMLU-Chem** asks multiple-choice chemistry questions from `cais/mmlu` and scores accuracy (random guessing = 25%).

- `use_rag=False` (default) measures what the model already knows.
- `use_rag=True` retrieves `k` chunks from every PDF in the `knowledge_source` folder for each question. The folder is indexed once and cached under `chroma_db/`; set `rebuild_index=True` after changing its contents.

Run it both ways to see how much your knowledge base helps.

## Usage

### Run the RAG pipeline

1. Put your PDFs in `Documents/` and set `CONFIG.documents.source`.
2. Run:

```bash
mamba activate rag_pipeline
python main.py
```

The first run embeds the source and caches the index; later runs on the same source start immediately. You then get an interactive prompt:

```text
Ask a question about the document. Type 'exit' to quit.

Question: What are the main findings?
```

### Run benchmarks

```bash
# Default: progress bar only
python benchmark_eval.py

# Verbose: show each question/answer
python benchmark_eval.py --verbose
```

Benchmark outputs are saved under:

```text
results/<benchmark>/<sanitized_model_name>/<timestamp>/
├── results.csv      # one row per question
└── summary.json     # aggregated metrics + full config snapshot
```

### Benchmark results

The JSON summary includes:

- Total, answerable, and unanswerable question counts.
- Exact-match rate and contains-answer rate.
- Hallucination rate on unanswerable questions.
- Average latency, total/average output tokens, and average tokens/second.
- The full `CONFIG` used for the run (for reproducibility).

## Notes

- The first run downloads the embedding model and (for benchmarks) the SQuAD v2 / MMLU datasets from Hugging Face.
- Ollama must be running if you use an Ollama model. The script will pull missing models automatically when `auto_pull=True`.
- The `results/` and `chroma_db/` directories are ignored by Git. `Documents/` is committed.

## License / attribution

Created using Ollama, VS Code, Mamba/Conda, and Claude AI.

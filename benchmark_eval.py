# ============================================================
# benchmark_eval.py
# Config-driven benchmark runner for the RAG pipeline.
# Supports multiple LLM providers and multiple benchmark datasets.
# Captures reasoning/thinking content when using thinking models.
# ============================================================

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import re
import shutil
import string
import sys
import time
from typing import Any

from datasets import load_dataset
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from config import (
    CONFIG,
    BenchmarksConfig,
    ChunkingConfig,
    EmbeddingConfig,
    ModelConfig,
    RetrievalConfig,
    SquadV2Config,
)
from main import build_rag_chain
from model_utils import create_llm, extract_response


# ------------------------------------------------------------
# Standard SQuAD-style text normalization for scoring
# (lowercase, strip punctuation/articles) — this is the
# official metric used by the SQuAD leaderboard itself.
# ------------------------------------------------------------
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = " ".join(s.split())
    return s


def exact_match(prediction: str, ground_truths: list[str]) -> bool | None:
    if not ground_truths:  # unanswerable case
        return None
    return any(normalize_answer(prediction) == normalize_answer(gt) for gt in ground_truths)


def contains_answer(prediction: str, ground_truths: list[str]) -> bool | None:
    """Looser check: did the answer appear anywhere in the response?"""
    if not ground_truths:
        return None
    pred_norm = normalize_answer(prediction)
    return any(normalize_answer(gt) in pred_norm for gt in ground_truths)


def looks_like_refusal(prediction: str) -> bool:
    """Checks if the model appropriately declined to answer."""
    refusal_phrases = [
        "don't have enough information",
        "cannot find",
        "not mentioned",
        "no information",
        "not provided",
        "don't know",
        "unable to answer",
    ]
    return any(phrase in prediction.lower() for phrase in refusal_phrases)


# ------------------------------------------------------------
# Results directory management
# ------------------------------------------------------------
def _sanitize_model_name(model_name: str) -> str:
    """Make a model name safe for use as a filesystem path component.

    Replaces characters that are problematic in file/directory names with
    underscores while keeping the name readable.
    """
    sanitized = re.sub(r"[:/\\|<>\"'?*]", "_", model_name)
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_")


def _prepare_results_dir(benchmark_name: str, model_name: str) -> str:
    """Create and return a timestamped results directory.

    Directory layout: results/<benchmark_name>/<sanitized_model_name>/<timestamp>/
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sanitized_model = _sanitize_model_name(model_name)
    results_dir = os.path.join(
        "results", benchmark_name, sanitized_model, timestamp
    )
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


# ------------------------------------------------------------
# Shared vector store builder for benchmarks
# ------------------------------------------------------------
def build_benchmark_store(
    contexts: list[str],
    chunking_config: ChunkingConfig,
    embedding_config: EmbeddingConfig,
    persist_directory: str = "benchmark_chroma_db",
) -> Chroma:
    """Build a Chroma vector store from a list of context strings."""
    # Remove any existing store so the embedding dimension matches the
    # configured model (e.g., when switching between small and large embeddings).
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunking_config.chunk_size,
        chunk_overlap=chunking_config.chunk_overlap,
    )
    docs = [Document(page_content=c) for c in contexts]
    chunks = splitter.split_documents(docs)

    embedding_model = HuggingFaceEmbeddings(model_name=embedding_config.model_name)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
    )
    return vector_store


# ------------------------------------------------------------
# Token/s measurement
# ------------------------------------------------------------
def _extract_token_stats(raw_response: Any, latency_seconds: float) -> dict[str, Any]:
    """Extract output token count and tokens/second from a model response.

    Falls back to latency-only if the provider does not expose token counts.
    """
    stats = {
        "output_tokens": None,
        "tokens_per_second": None,
        "latency_seconds": latency_seconds,
    }

    if not isinstance(raw_response, AIMessage):
        return stats

    metadata = raw_response.response_metadata or {}

    # Ollama exposes eval_count (tokens) and eval_duration (nanoseconds).
    if "eval_count" in metadata and "eval_duration" in metadata:
        output_tokens = metadata["eval_count"]
        duration_ns = metadata["eval_duration"]
        if duration_ns and duration_ns > 0:
            stats["output_tokens"] = output_tokens
            stats["tokens_per_second"] = output_tokens / (duration_ns / 1e9)
        else:
            stats["output_tokens"] = output_tokens
        return stats

    # OpenAI-compatible APIs expose token_usage with completion_tokens.
    token_usage = metadata.get("token_usage") or metadata.get("usage")
    if isinstance(token_usage, dict):
        output_tokens = token_usage.get("completion_tokens") or token_usage.get(
            "output_tokens"
        )
        if output_tokens:
            stats["output_tokens"] = output_tokens
            if latency_seconds > 0:
                stats["tokens_per_second"] = output_tokens / latency_seconds
        return stats

    # Anthropic exposes usage.output_tokens.
    usage = metadata.get("usage")
    if isinstance(usage, dict):
        output_tokens = usage.get("output_tokens")
        if output_tokens:
            stats["output_tokens"] = output_tokens
            if latency_seconds > 0:
                stats["tokens_per_second"] = output_tokens / latency_seconds
        return stats

    return stats


# ------------------------------------------------------------
# Benchmark: SQuAD v2
# ------------------------------------------------------------
def run_squad_v2(
    benchmark_config: SquadV2Config,
    model_config: ModelConfig,
    chunking_config: ChunkingConfig,
    embedding_config: EmbeddingConfig,
    retrieval_config: RetrievalConfig,
    llm,
    verbosity: int = 0,
    results_dir: str | None = None,
) -> list[dict]:
    """Run the SQuAD v2 benchmark and return per-question results."""
    print("Loading SQuAD v2 sample...")
    dataset = load_dataset(benchmark_config.dataset_name, split=benchmark_config.split)

    # Take a manageable sample: mix of answerable + unanswerable
    sample = dataset.shuffle(seed=benchmark_config.seed).select(
        range(benchmark_config.num_questions)
    )

    # Collect unique contexts to build the vector store from
    unique_contexts = list(set(sample["context"]))
    print(f"Building vector store from {len(unique_contexts)} unique passages...")
    vector_store = build_benchmark_store(
        contexts=unique_contexts,
        chunking_config=chunking_config,
        embedding_config=embedding_config,
        persist_directory="benchmark_chroma_db",
    )

    # Build chain WITHOUT an output parser so we can inspect the raw AIMessage
    # and extract thinking/reasoning content.
    chain = build_rag_chain(
        vector_store=vector_store,
        llm=llm,
        retrieval_config=retrieval_config,
        output_parser=None,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": retrieval_config.k})

    results: list[dict] = []
    total = len(sample)

    # Choose iteration style based on verbosity.
    if verbosity == 0:
        iterator = tqdm(enumerate(sample), total=total, desc="SQuAD v2")
    else:
        iterator = enumerate(sample)

    for i, item in iterator:
        question = item["question"]
        ground_truths = item["answers"]["text"]  # empty list = unanswerable

        # Get the retrieved chunks separately so we can inspect them
        retrieved_docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in retrieved_docs]

        if verbosity >= 1:
            print(f"\nQ {i + 1}/{total}: {question}")

        start_time = time.perf_counter()
        raw_response = chain.invoke(question)
        latency_seconds = time.perf_counter() - start_time

        response_parts = extract_response(raw_response, model_config)
        final_answer = response_parts["final_answer"]
        token_stats = _extract_token_stats(raw_response, latency_seconds)

        if verbosity >= 1:
            print(f"A: {final_answer}")
            if response_parts["thinking"]:
                print("Thinking captured (truncated):")
                print(response_parts["thinking"][:500] + "...")

        # Debug print for just the first couple of questions (verbosity >= 2).
        if verbosity >= 2 and i < 2:
            print("--- DEBUG ---")
            print(f"Ground truth: {ground_truths}")
            print(f"Retrieved chunks:\n{[c[:150] for c in contexts]}")

        em = exact_match(final_answer, ground_truths)
        contains = contains_answer(final_answer, ground_truths)
        refused = looks_like_refusal(final_answer)
        hallucinated = (not ground_truths) and (not refused)

        results.append(
            {
                "question": question,
                "answerable": bool(ground_truths),
                "raw_answer": response_parts["raw_answer"],
                "thinking": response_parts["thinking"],
                "final_answer": final_answer,
                "exact_match": em,
                "contains_answer": contains,
                "refused": refused,
                "hallucinated": hallucinated,
                "latency_seconds": token_stats["latency_seconds"],
                "output_tokens": token_stats["output_tokens"],
                "tokens_per_second": token_stats["tokens_per_second"],
            }
        )

    # Save detailed per-question results under the timestamped results directory.
    output_dir = results_dir or "."
    output_file = os.path.join(output_dir, "results.csv")
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nDetailed results saved to {output_file}")

    return results


def _print_squad_v2_summary(
    results: list[dict],
    results_dir: str | None = None,
    config_obj: Any | None = None,
) -> None:
    """Print summary statistics for SQuAD v2 results and save them as JSON."""
    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]

    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS: SQuAD v2")
    print("=" * 50)
    print(f"Total questions: {len(results)}")
    print(f"Answerable questions: {len(answerable)}")
    print(f"Unanswerable questions: {len(unanswerable)}")

    summary: dict[str, Any] = {
        "benchmark": "squad_v2",
        "total_questions": len(results),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
    }

    if answerable:
        em_rate = sum(1 for r in answerable if r["exact_match"]) / len(answerable)
        contains_rate = sum(1 for r in answerable if r["contains_answer"]) / len(answerable)
        print(f"\nExact match rate: {em_rate:.1%}")
        print(f"Contains-answer rate: {contains_rate:.1%}")
        summary["exact_match_rate"] = em_rate
        summary["contains_answer_rate"] = contains_rate

    if unanswerable:
        hallucination_rate = sum(1 for r in unanswerable if r["hallucinated"]) / len(
            unanswerable
        )
        print(
            f"\nHallucination rate on unanswerable questions: {hallucination_rate:.1%}"
        )
        print(
            "(Lower is better — this is the model making things up when it shouldn't)"
        )
        summary["hallucination_rate"] = hallucination_rate

    # Token/s and latency statistics.
    latencies = [r["latency_seconds"] for r in results if r["latency_seconds"] is not None]
    token_counts = [r["output_tokens"] for r in results if r["output_tokens"] is not None]
    token_rates = [r["tokens_per_second"] for r in results if r["tokens_per_second"] is not None]

    performance: dict[str, Any] = {}
    print("\n--- Performance ---")
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        print(f"Average latency: {avg_latency:.3f}s")
        performance["avg_latency_seconds"] = avg_latency
    if token_counts:
        total_tokens = sum(token_counts)
        avg_tokens = total_tokens / len(token_counts)
        print(f"Total output tokens: {total_tokens}")
        print(f"Average output tokens: {avg_tokens:.1f}")
        performance["total_output_tokens"] = total_tokens
        performance["avg_output_tokens"] = avg_tokens
    if token_rates:
        avg_tps = sum(token_rates) / len(token_rates)
        print(f"Average tokens/second: {avg_tps:.1f}")
        performance["avg_tokens_per_second"] = avg_tps
    if not any([latencies, token_counts, token_rates]):
        print("No performance metrics available for this provider.")

    if performance:
        summary["performance"] = performance

    # Embed the full configuration for reproducibility.
    if config_obj is not None:
        summary["config"] = config_obj.model_dump()

    # Save summary to JSON under the results directory.
    output_dir = results_dir or "."
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to {summary_file}")


# ------------------------------------------------------------
# Benchmark dispatch
# ------------------------------------------------------------
BENCHMARK_DISPATCH = {
    "squad_v2": (run_squad_v2, _print_squad_v2_summary),
}


def run_benchmarks(
    benchmarks_config: BenchmarksConfig,
    model_config: ModelConfig,
    chunking_config: ChunkingConfig,
    embedding_config: EmbeddingConfig,
    retrieval_config: RetrievalConfig,
    llm,
    verbosity: int = 0,
    config_obj: Any | None = None,
) -> None:
    """Run all benchmarks listed in the enabled config."""
    for benchmark_name in benchmarks_config.enabled:
        if benchmark_name not in BENCHMARK_DISPATCH:
            print(
                f"Warning: Unknown benchmark '{benchmark_name}'. "
                f"Supported: {list(BENCHMARK_DISPATCH.keys())}. Skipping.",
                file=sys.stderr,
            )
            continue

        print(f"\n{'=' * 50}")
        print(f"Running benchmark: {benchmark_name}")
        print("=" * 50)

        # Create a timestamped results directory keyed by benchmark/model.
        results_dir = _prepare_results_dir(benchmark_name, model_config.model)
        print(f"Results will be saved to: {results_dir}")

        runner, summary_printer = BENCHMARK_DISPATCH[benchmark_name]
        benchmark_params = getattr(benchmarks_config, benchmark_name)

        results = runner(
            benchmark_config=benchmark_params,
            model_config=model_config,
            chunking_config=chunking_config,
            embedding_config=embedding_config,
            retrieval_config=retrieval_config,
            llm=llm,
            verbosity=verbosity,
            results_dir=results_dir,
        )

        summary_printer(results, results_dir=results_dir, config_obj=config_obj)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run RAG benchmarks with configurable verbosity."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (show each question/answer with Q x/total progress).",
    )
    return parser.parse_args()


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    args = _parse_args()
    verbosity = 1 if args.verbose else CONFIG.benchmarks.verbosity

    llm = create_llm(CONFIG.model)

    run_benchmarks(
        benchmarks_config=CONFIG.benchmarks,
        model_config=CONFIG.model,
        chunking_config=CONFIG.chunking,
        embedding_config=CONFIG.embedding,
        retrieval_config=CONFIG.retrieval,
        llm=llm,
        verbosity=verbosity,
        config_obj=CONFIG,
    )

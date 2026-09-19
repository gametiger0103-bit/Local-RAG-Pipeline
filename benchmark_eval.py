# ============================================================
# benchmark_eval.py
# Tests the RAG pipeline against a subset of SQuAD v2 —
# a standard public QA benchmark with ground-truth answers
# AND unanswerable questions (for hallucination testing).
# ============================================================

import os
import re
import string
from datasets import load_dataset

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from main import build_rag_chain

# ------------------------------------------------------------
# Standard SQuAD-style text normalization for scoring
# (lowercase, strip punctuation/articles) — this is the
# official metric used by the SQuAD leaderboard itself.
# ------------------------------------------------------------
def normalize_answer(s):
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = ' '.join(s.split())
    return s

def exact_match(prediction, ground_truths):
    if not ground_truths:  # unanswerable case
        return None
    return any(normalize_answer(prediction) == normalize_answer(gt) for gt in ground_truths)

def contains_answer(prediction, ground_truths):
    """Looser check: did the answer appear anywhere in the response?"""
    if not ground_truths:
        return None
    pred_norm = normalize_answer(prediction)
    return any(normalize_answer(gt) in pred_norm for gt in ground_truths)

def looks_like_refusal(prediction):
    """Checks if the model appropriately declined to answer."""
    refusal_phrases = ["don't have enough information", "cannot find", "not mentioned",
                        "no information", "not provided", "don't know", "unable to answer"]
    return any(phrase in prediction.lower() for phrase in refusal_phrases)

# ------------------------------------------------------------
# Build a temporary vector store from SQuAD context paragraphs
# ------------------------------------------------------------
def build_benchmark_store(contexts, persist_directory="benchmark_chroma_db"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = [Document(page_content=c) for c in contexts]
    chunks = splitter.split_documents(docs)

    embedding_model = embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    return vector_store

if __name__ == "__main__":
    print("Loading SQuAD v2 sample...")
    dataset = load_dataset("rajpurkar/squad_v2", split="validation")

    # Take a manageable sample: mix of answerable + unanswerable
    sample = dataset.shuffle(seed=42).select(range(30))

    # Collect unique contexts to build the vector store from
    unique_contexts = list(set(sample["context"]))
    print(f"Building vector store from {len(unique_contexts)} unique passages...")
    vector_store = build_benchmark_store(unique_contexts)

    chain = build_rag_chain(vector_store)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    results = []
    for i, item in enumerate(sample):
        question = item["question"]
        ground_truths = item["answers"]["text"]  # empty list = unanswerable

        # Get the retrieved chunks separately so we can inspect them
        retrieved_docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in retrieved_docs]

        print(f"\nQ: {question}")
        answer = chain.invoke(question)
        print(f"A: {answer}")

        # Debug print for just the first couple of questions
        if i < 2:
            print(f"--- DEBUG ---")
            print(f"Ground truth: {ground_truths}")
            print(f"Retrieved chunks:\n{[c[:150] for c in contexts]}")

        em = exact_match(answer, ground_truths)
        contains = contains_answer(answer, ground_truths)
        refused = looks_like_refusal(answer)

        hallucinated = (not ground_truths) and (not refused)

        results.append({
            "question": question,
            "answerable": bool(ground_truths),
            "exact_match": em,
            "contains_answer": contains,
            "refused": refused,
            "hallucinated": hallucinated
        })

    # ------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------
    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]

    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Total questions: {len(results)}")
    print(f"Answerable questions: {len(answerable)}")
    print(f"Unanswerable questions: {len(unanswerable)}")

    if answerable:
        em_rate = sum(1 for r in answerable if r["exact_match"]) / len(answerable)
        contains_rate = sum(1 for r in answerable if r["contains_answer"]) / len(answerable)
        print(f"\nExact match rate: {em_rate:.1%}")
        print(f"Contains-answer rate: {contains_rate:.1%}")

    if unanswerable:
        hallucination_rate = sum(1 for r in unanswerable if r["hallucinated"]) / len(unanswerable)
        print(f"\nHallucination rate on unanswerable questions: {hallucination_rate:.1%}")
        print("(Lower is better — this is the model making things up when it shouldn't)")

    import csv
    with open("benchmark_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print("\nDetailed results saved to benchmark_results.csv")

 
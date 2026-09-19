# ============================================================
# mmlu_chem_eval.py
# Tests the base LLM (no retrieval) on MMLU-Chem —
# a standard multiple-choice chemistry knowledge benchmark.
# This measures the model's raw chemistry knowledge,
# separate from your RAG retrieval pipeline.
# ============================================================

import re
import csv
from datasets import load_dataset
from langchain_ollama import ChatOllama

LETTERS = ["A", "B", "C", "D"]

def format_question(item):
    """Builds a strict multiple-choice prompt from one dataset item."""
    question = item["question"]
    choices = item["choices"]

    options_text = "\n".join(f"{LETTERS[i]}. {choice}" for i, choice in enumerate(choices))

    prompt = f"""You are a chemistry expert taking a multiple choice exam.
Answer the following question by responding with ONLY the letter of the
correct option (A, B, C, or D). Do not explain your reasoning. Do not
include any other text.

Question: {question}

{options_text}

Answer:"""
    return prompt

def extract_letter(response):
    """
    Pulls the first standalone A/B/C/D letter out of the model's
    response. Models often add extra words despite instructions,
    so this looks for the first clear letter match.
    """
    match = re.search(r'\b([A-D])\b', response.strip().upper())
    if match:
        return match.group(1)
    return None

if __name__ == "__main__":
    print("Loading MMLU-Chem (college_chemistry) sample...")
    dataset = load_dataset("cais/mmlu", "college_chemistry", split="test")

    # Keep the sample small for a first run — bump this up once it's working
    sample = dataset.shuffle(seed=42).select(range(20))

    llm = ChatOllama(model="llama3.2:1b", temperature=0, repeat_penalty=1.3)

    results = []
    correct_count = 0

    for i, item in enumerate(sample):
        prompt = format_question(item)
        correct_letter = LETTERS[item["answer"]]  # answer key: 0-3 -> A-D

        print(f"\nQ{i+1}: {item['question'][:80]}...")
        try:
            response = llm.invoke(prompt).content
        except Exception as e:
            print(f"  ⚠ Generation failed: {e}")
            response = ""
        predicted_letter = extract_letter(response)

        is_correct = (predicted_letter == correct_letter)
        if is_correct:
            correct_count += 1

        print(f"Predicted: {predicted_letter} | Correct: {correct_letter} | {'✓' if is_correct else '✗'}")

        results.append({
            "question": item["question"],
            "correct_answer": correct_letter,
            "predicted_answer": predicted_letter,
            "raw_response": response.strip(),
            "is_correct": is_correct
        })

    accuracy = correct_count / len(sample)

    print("\n" + "=" * 50)
    print("MMLU-CHEM BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Total questions: {len(sample)}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.1%}")

    with open("mmlu_chem_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print("\nDetailed results saved to mmlu_chem_results.csv")
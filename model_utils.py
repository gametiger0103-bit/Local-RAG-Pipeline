# ============================================================
# model_utils.py
# Helpers for instantiating LLMs and handling model-specific
# response formats, including thinking/reasoning extraction and
# Ollama model auto-pulling.
# ============================================================

from __future__ import annotations

import re
import sys
from typing import Any

from langchain_core.messages import AIMessage
from tqdm import tqdm

from config import ModelConfig


def _normalize_model_name(name: str) -> str:
    """Normalize an Ollama model name for comparison.

    Ollama stores names like ``llama3.2:1b``; users may type
    ``llama3.2`` (which resolves to ``llama3.2:latest``). We keep
    the exact configured tag to avoid surprises, but strip redundant
    ``:latest`` only when comparing.
    """
    return name if not name.endswith(":latest") else name[: -len(":latest")]


def ensure_ollama_model(model_name: str, auto_pull: bool = True) -> None:
    """Ensure an Ollama model is available locally, pulling it if needed.

    Args:
        model_name: The Ollama model name (e.g. ``llama3.2:1b``).
        auto_pull: If True and the model is missing, pull it automatically.

    Raises:
        SystemExit: If the model is missing and cannot be pulled, or if the
            pull fails (e.g., typo, network error).
    """
    import ollama

    try:
        local_models = ollama.list().get("models", [])
    except Exception as exc:  # pragma: no cover
        print(f"Error: Could not list local Ollama models: {exc}", file=sys.stderr)
        sys.exit(1)

    local_names = {_normalize_model_name(m.get("name", m.get("model", ""))) for m in local_models}

    if _normalize_model_name(model_name) in local_names:
        print(f"Ollama model '{model_name}' is already available locally.")
        return

    if not auto_pull:
        print(
            f"Error: Ollama model '{model_name}' is not available locally and "
            f"auto_pull is disabled.",
            file=sys.stderr,
        )
        print(
            f"Run `ollama pull {model_name}` manually and try again.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Ollama model '{model_name}' not found locally. Pulling now...")

    try:
        progress_bar: tqdm | None = None
        for progress in ollama.pull(model_name, stream=True):
            status = progress.get("status", "")
            completed = progress.get("completed")
            total = progress.get("total")

            if status == "success":
                if progress_bar is not None:
                    progress_bar.close()
                print(f"Successfully pulled '{model_name}'.")
                return

            if total is not None and total > 0:
                if progress_bar is None:
                    progress_bar = tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        desc=f"Pulling {model_name}",
                    )
                if completed is not None:
                    progress_bar.n = completed
                    progress_bar.total = total
                    progress_bar.refresh()

        # If the stream ended without an explicit success event, close the bar.
        if progress_bar is not None:
            progress_bar.close()

    except ollama.ResponseError as exc:
        sys.stdout.flush()
        error_msg = str(exc).lower()
        is_not_found = (
            exc.status_code == 404
            or exc.status_code == -1
            or "file does not exist" in error_msg
            or "not found" in error_msg
        )
        if is_not_found:
            print(
                f"\nError: Model '{model_name}' was not found on Ollama.",
                file=sys.stderr,
            )
            print(
                "Please check the model name at https://ollama.com/library",
                file=sys.stderr,
            )
            print(
                "or run `ollama pull <correct-model-name>` manually.",
                file=sys.stderr,
            )
        else:
            print(
                f"\nError: Failed to pull Ollama model '{model_name}': {exc}",
                file=sys.stderr,
            )
        sys.exit(1)
    except Exception as exc:  # pragma: no cover
        sys.stdout.flush()
        print(
            f"\nError: Unexpected error while pulling '{model_name}': {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


def create_llm(config: ModelConfig) -> Any:
    """Create a LangChain chat model from the provided configuration.

    Args:
        config: The model configuration.

    Returns:
        A LangChain chat model instance.
    """
    if config.provider == "ollama":
        from langchain_ollama import ChatOllama

        ensure_ollama_model(config.model, auto_pull=config.auto_pull)
        return ChatOllama(model=config.model, **config.parameters)

    if config.provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=config.model, **config.parameters)

    if config.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=config.model, **config.parameters)

    if config.provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=config.model, **config.parameters)

    raise ValueError(f"Unsupported model provider: {config.provider}")


def _extract_thinking_ollama(content: str) -> tuple[str, str]:
    """Extract content inside <think>...</think> tags used by some Ollama
    reasoning models (e.g., DeepSeek-R1, QwQ)."""
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
    thinking_parts = pattern.findall(content)
    if thinking_parts:
        thinking = "\n\n".join(part.strip() for part in thinking_parts)
        final_answer = pattern.sub("", content).strip()
        return thinking, final_answer
    return "", content.strip()


def extract_response(response: AIMessage | str, config: ModelConfig) -> dict[str, Any]:
    """Extract raw answer, reasoning/thinking content, and final answer.

    Args:
        response: The raw model response (AIMessage or string).
        config: The model configuration.

    Returns:
        A dictionary with keys:
            - ``raw_answer``: the full raw text from the model
            - ``thinking``: reasoning/thinking content (empty for non-thinking models)
            - ``final_answer``: the answer to be used for downstream evaluation
    """
    if isinstance(response, str):
        raw = response
    else:
        raw = response.content
        if not isinstance(raw, str):
            raw = str(raw)

    thinking = ""
    final_answer = raw.strip()

    if not config.is_thinking_model:
        return {"raw_answer": raw, "thinking": "", "final_answer": final_answer}

    # Provider-specific thinking extraction.
    if config.provider == "openai":
        # o1/o3 series expose reasoning_content in additional_kwargs.
        thinking = ""
        if isinstance(response, AIMessage):
            thinking = response.additional_kwargs.get("reasoning_content", "")
            if thinking:
                final_answer = raw.strip()

    elif config.provider == "anthropic":
        if isinstance(response, AIMessage):
            # Claude thinking content may appear in response_metadata.
            thinking_blocks = response.response_metadata.get("thinking", [])
            if thinking_blocks:
                if isinstance(thinking_blocks, list):
                    thinking = "\n\n".join(
                        b.get("thinking", "") if isinstance(b, dict) else str(b)
                        for b in thinking_blocks
                    )
                else:
                    thinking = str(thinking_blocks)

    elif config.provider == "google":
        if isinstance(response, AIMessage):
            thinking = response.additional_kwargs.get("thought", "")
            if not thinking:
                thinking = response.response_metadata.get("thought", "")

    elif config.provider == "ollama":
        thinking, final_answer = _extract_thinking_ollama(raw)

    return {
        "raw_answer": raw,
        "thinking": thinking.strip() if thinking else "",
        "final_answer": final_answer.strip(),
    }

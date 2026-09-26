# ============================================================
# RAG PIPELINE - main.py
# Local RAG pipeline configurable via config.py.
# Supports multiple LLM providers (Ollama, OpenAI, Anthropic, Google).
# ============================================================

import hashlib
import os
import re
import shutil
import sys
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CONFIG,
    ChunkingConfig,
    DocumentsConfig,
    EmbeddingConfig,
    RetrievalConfig,
)
from model_utils import create_llm


def store_path_for(
    documents_config: DocumentsConfig | None = None,
    embedding_config: EmbeddingConfig | None = None,
):
    """Maps a document source to its own vector store directory.

    The directory name hashes the absolute source path and the embedding
    model, so same-named files in different folders stay separate and
    switching embedding models never reuses an incompatible index.
    """
    if documents_config is None:
        documents_config = CONFIG.documents
    if embedding_config is None:
        embedding_config = CONFIG.embedding

    path = Path(documents_config.source).resolve()
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", path.stem).strip("-").lower() or "docs"
    key = f"{path}|{embedding_config.model_name}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:8]
    return os.path.join(documents_config.persist_root, f"{slug}-{digest}")


def load_documents(source: str):
    """Loads a single PDF file, or all PDFs in a directory."""
    path = Path(source)

    if not path.exists():
        sys.exit(f"Error: no such file or folder: {source}")

    if path.is_file():
        if path.suffix.lower() != ".pdf":
            sys.exit(f"Error: not a PDF file: {source}")
        print(f"Loading document: {path}")
        loader = PyPDFLoader(str(path))
    else:
        print(f"Loading documents from: {path}")
        loader = DirectoryLoader(str(path), glob="**/*.pdf", loader_cls=PyPDFLoader)

    documents = loader.load()
    if not documents:
        sys.exit(f"Error: no PDF content found in: {source}")

    print(f"Loaded {len(documents)} pages")
    return documents


def chunk_documents(documents, chunking_config: ChunkingConfig | None = None):
    """Splits documents into smaller overlapping chunks."""
    if chunking_config is None:
        chunking_config = CONFIG.chunking

    print("Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunking_config.chunk_size,
        chunk_overlap=chunking_config.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks


def build_vector_store(
    chunks,
    persist_directory: str = "chroma_db",
    embedding_config: EmbeddingConfig | None = None,
):
    """Embeds chunks and stores them in ChromaDB."""
    if embedding_config is None:
        embedding_config = CONFIG.embedding

    print("Building vector store...")
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_config.model_name)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
    )
    print(f"Vector store saved to: {persist_directory}")
    return vector_store


def load_vector_store(
    persist_directory: str = "chroma_db",
    embedding_config: EmbeddingConfig | None = None,
):
    """Loads existing vector store from disk."""
    if embedding_config is None:
        embedding_config = CONFIG.embedding

    print("Loading existing vector store...")
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_config.model_name)
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
    )
    return vector_store


def build_rag_chain(
    vector_store,
    llm,
    retrieval_config: RetrievalConfig | None = None,
    output_parser=StrOutputParser(),
):
    """Builds the RAG chain connecting retriever to the provided LLM.

    Args:
        vector_store: The Chroma vector store to retrieve from.
        llm: The LangChain chat model to use for generation.
        retrieval_config: Retriever configuration. Defaults to CONFIG.retrieval.
        output_parser: Optional output parser. Pass ``None`` to receive the raw
            ``AIMessage`` (useful for capturing thinking/reasoning content).

    Returns:
        The assembled LCEL chain.
    """
    if retrieval_config is None:
        retrieval_config = CONFIG.retrieval

    print("Building RAG chain...")

    prompt = PromptTemplate.from_template("""
You are a helpful assistant. Use the following context to answer the question.
Only answer using the context above. If the context does not contain the answer, respond with exactly: "I don't have enough information to answer that." Do not guess.

Context:
{context}

Question: {question}

Answer:""")

    retriever = vector_store.as_retriever(search_kwargs={"k": retrieval_config.k})

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    if output_parser is not None:
        rag_chain = rag_chain | output_parser

    return rag_chain


def ask(chain, question: str):
    """Runs a question through the RAG chain and prints the answer."""
    print(f"\nQuestion: {question}")
    answer = chain.invoke(question)
    print(f"Answer: {answer}")
    return answer


def chat_loop(chain):
    """Prompts for questions until the user exits."""
    print("\nAsk a question about the document. Type 'exit' to quit.\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        try:
            answer = chain.invoke(question)
        except Exception as e:
            print(f"Error: {e}\n")
            continue

        print(f"\nAnswer: {answer}\n")


if __name__ == "__main__":
    db_path = store_path_for()

    # Instantiate the configured LLM (will auto-pull Ollama models if needed).
    llm = create_llm(CONFIG.model)

    if CONFIG.documents.rebuild and os.path.exists(db_path):
        print(f"Rebuilding index at: {db_path}")
        shutil.rmtree(db_path)

    if os.path.exists(db_path):
        vector_store = load_vector_store(db_path)
    else:
        documents = load_documents(CONFIG.documents.source)
        chunks = chunk_documents(documents)
        vector_store = build_vector_store(chunks, db_path)

    chain = build_rag_chain(vector_store, llm)
    chat_loop(chain)

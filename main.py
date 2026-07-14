# ============================================================
# RAG PIPELINE - main.py
# Local RAG pipeline using Ollama (no API key needed)
# ============================================================

import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def load_documents(directory: str):
    """Loads all PDF files from the specified directory."""
    print(f"Loading documents from: {directory}")
    loader = DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")
    return documents

def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    """Splits documents into smaller overlapping chunks."""
    print("Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks

def build_vector_store(chunks, persist_directory="chroma_db"):
    """Embeds chunks and stores them in ChromaDB."""
    print("Building vector store...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    print(f"Vector store saved to: {persist_directory}")
    return vector_store

def load_vector_store(persist_directory="chroma_db"):
    """Loads existing vector store from disk."""
    print("Loading existing vector store...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )
    return vector_store

def build_rag_chain(vector_store):
    """Builds the RAG chain connecting retriever to local Ollama LLM."""
    print("Building RAG chain...")

    # Local LLM via Ollama - no API key needed
    llm = ChatOllama(
        model="llama3.2:1b",
        temperature=0
    )

    prompt = PromptTemplate.from_template("""
You are a helpful assistant. Use the following context to answer the question.
If you don't know the answer from the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:""")

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def ask(chain, question: str):
    """Runs a question through the RAG chain and prints the answer."""
    print(f"\nQuestion: {question}")
    answer = chain.invoke(question)
    print(f"Answer: {answer}")
    return answer

if __name__ == "__main__":
    db_path = "chroma_db"

    if not os.path.exists(db_path):
        documents = load_documents("documents")
        chunks = chunk_documents(documents)
        vector_store = build_vector_store(chunks, db_path)
    else:
        vector_store = load_vector_store(db_path)

import time

chain = build_rag_chain(vector_store)

ask(chain, "What is this document about?")
time.sleep(3)  # Give Ollama a moment to reset between calls

ask(chain, "What are the main findings?")
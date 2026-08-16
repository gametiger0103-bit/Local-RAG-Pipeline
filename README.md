# Local RAG Pipeline

Overview
This project is a Retrieval-Augmented Generation (RAG) pipeline that uses:
- Ollama
- LangChain
- ChromaDB
- Python

## Features
- Loads PDF documents
- Splits documents into chunks
- Generates embeddings
- Stores vectors in ChromaDB
- Retrieves relevant context
- Uses Ollama to answer questions grounded in the documents

# Created using
 - Ollama
 - VS Code
 - Conda
 - Claude AI

## Installation

```bash
conda env create -f environment.yml
conda activate <your-environment-name>

# HOW TO USE:
1. conda activate rag_pipeline
2. rmdir /s /q chroma_db
3. python main.py

**WIP**
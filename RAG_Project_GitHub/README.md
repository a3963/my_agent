# RAG Project

A retrieval-augmented generation project with document loading, vector retrieval, reranking, answer generation, and evaluation scripts.

## Local setup

1. Create a Python virtual environment and install the dependencies in `requirements.txt`.
2. Copy `.env.example` to `.env`.
3. Put your own documents in `data/documents/`.
4. Set `OPENAI_API_KEY` in `.env`; never commit that file.
5. Build the local knowledge base and start the application using the scripts in `backend/`.

## Repository safety

The repository intentionally excludes API keys, local environment files, IDE settings, Python caches, source documents, and generated vector databases.

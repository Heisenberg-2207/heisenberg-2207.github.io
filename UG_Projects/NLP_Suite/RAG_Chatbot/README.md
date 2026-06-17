# RAG PDF Chatbot

> Independent project · IIT Madras · 2025

Fully offline Retrieval-Augmented Generation (RAG) system for querying multiple PDFs using a local LLM. No cloud API required.

## What's Here

```
RAG_Chatbot/
├── rag_system.py       ← Core RAG engine (FAISS + embeddings + Ollama)
├── chatbot.py          ← CLI chatbot interface
├── main.ipynb          ← Interactive demo notebook
├── documents/          ← Folder for your PDF files
├── embeddings.npy      ← Cached embeddings (auto-generated)
├── index.faiss         ← FAISS vector index (auto-generated)
└── metadata.pkl        ← Chunk metadata (auto-generated)
```

## Architecture

```
PDF documents
     ↓
Token-based chunking (with overlap for context continuity)
     ↓
MiniLM sentence embeddings (all-MiniLM-L6-v2)
     ↓
FAISS index (cosine similarity)
     ↓
Query → retrieve top-k chunks
     ↓
Local LLM via Ollama (Mistral / Llama2 / etc.)
     ↓
Answer with document name + page number citations
```

## Requirements

```bash
pip install pdfplumber tiktoken numpy faiss-cpu sentence-transformers
```

Install [Ollama](https://ollama.com) for local LLM inference:
```bash
ollama pull mistral
```

## Usage

```python
from rag_system import RAGSystem

# Load PDFs and build FAISS index (cached after first run)
rag = RAGSystem(folder_path="documents/")

# Ask questions
answer = rag.answer_query("What are Aniket's research interests?")
print(answer)
# → "Based on Aniket_CV.pdf (page 1): Aniket's research interests include..."
```

## Features

- **Persistent cache:** embeddings, FAISS index, and metadata saved to disk
- **Multi-document:** handles an entire folder of PDFs
- **Page citations:** every answer includes the source document and page
- **Fully offline:** no OpenAI, no Anthropic, no internet after setup
- **Adjustable top-k:** `answer_query(query, top_k=5)` for more/less context

## Performance Notes

- MiniLM is fast and small; adequate for personal-scale document sets
- For large corpora (>1,000 pages), consider `all-mpnet-base-v2` for better recall
- FAISS cosine search is O(n) but extremely fast in practice (<1 ms for <10K chunks)

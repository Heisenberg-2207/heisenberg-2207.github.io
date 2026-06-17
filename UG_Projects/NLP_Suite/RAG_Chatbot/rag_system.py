# rag_system_optimized.py
import os
import glob
import pdfplumber
import tiktoken
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import subprocess

# -------------------------------
# Tokenizer setup
# -------------------------------
tokenizer = tiktoken.get_encoding("cl100k_base")

# -------------------------------
# PDF to chunks
# -------------------------------
def pdf_to_chunks(pdf_path, chunk_size=500, overlap=50):
    chunks = []
    doc_name = os.path.basename(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            tokens = tokenizer.encode(text)

            start = 0
            while start < len(tokens):
                end = start + chunk_size
                chunk_tokens = tokens[start:end]
                chunk_text = tokenizer.decode(chunk_tokens)

                chunks.append({
                    "document": doc_name,
                    "page": i,
                    "chunk": chunk_text,
                    "tokens": len(chunk_tokens)
                })

                start += chunk_size - overlap

    return chunks

def folder_to_chunks(folder_path, chunk_size=500, overlap=50):
    all_chunks = []
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))

    for pdf_file in pdf_files:
        print(f"Processing: {os.path.basename(pdf_file)}")
        chunks = pdf_to_chunks(pdf_file, chunk_size, overlap)
        all_chunks.extend(chunks)

    return all_chunks

# -------------------------------
# Embeddings and FAISS
# -------------------------------
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text):
    return embedder.encode([text])[0].astype("float32")

def build_faiss_index(chunks, index_file="index.faiss", metadata_file="metadata.pkl", embeddings_file="embeddings.npy"):
    """
    Build or load FAISS index with caching.
    """
    # Check if files exist
    if os.path.exists(index_file) and os.path.exists(metadata_file) and os.path.exists(embeddings_file):
        print("Loading cached FAISS index and metadata...")
        index = faiss.read_index(index_file)
        with open(metadata_file, "rb") as f:
            metadata = pickle.load(f)
        embeddings = np.load(embeddings_file)
        return index, metadata, embeddings

    # Otherwise, build index from scratch
    print("Building FAISS index from scratch...")
    dim = len(get_embedding("test"))
    index = faiss.IndexFlatL2(dim)
    embeddings = []
    metadata = []

    for ch in chunks:
        emb = get_embedding(ch["chunk"])
        embeddings.append(emb)
        metadata.append(ch)

    embeddings = np.vstack(embeddings)
    index.add(embeddings)

    # Save for future use
    faiss.write_index(index, index_file)
    with open(metadata_file, "wb") as f:
        pickle.dump(metadata, f)
    np.save(embeddings_file, embeddings)

    return index, metadata, embeddings

def query_faiss(query, index, metadata, embeddings=None, k=3):
    """
    Query FAISS index and return top-k relevant chunks.
    """
    q_emb = get_embedding(query).reshape(1, -1)
    D, I = index.search(q_emb, k)
    results = [metadata[idx] for idx in I[0]]
    return results

# -------------------------------
# Ollama LLM
# -------------------------------
def ask_ollama(prompt, model="mistral", ollama_path="ollama"):
    """
    Run inference using Ollama installed locally.
    """
    result = subprocess.run(
        [ollama_path, "run", model],
        input=prompt.encode("utf-8"),
        capture_output=True
    )
    return result.stdout.decode("utf-8")

# -------------------------------
# Full query answering
# -------------------------------
class RAGSystem:
    def __init__(self, folder_path="documents", chunk_size=400, overlap=50,
                 index_file="index.faiss", metadata_file="metadata.pkl", embeddings_file="embeddings.npy"):
        print("Loading PDFs and creating chunks...")
        self.chunks = folder_to_chunks(folder_path, chunk_size, overlap)
        print(f"Total chunks: {len(self.chunks)}")

        print("Building or loading FAISS index...")
        self.index, self.metadata, self.embeddings = build_faiss_index(
            self.chunks,
            index_file=index_file,
            metadata_file=metadata_file,
            embeddings_file=embeddings_file
        )
        print("RAG system ready.")

    def answer_query(self, query, top_k=3, model="mistral", ollama_path="ollama"):
        top_chunks = query_faiss(query, self.index, self.metadata, k=top_k)

        context = "\n\n".join(
            [f"[Document: {c['document']} - Page: {c['page']}] {c['chunk']}" for c in top_chunks]
        )

        final_prompt = f"""
        Answer the question using the following context. 
        Explicitly mention the document name and page number for any referenced information.

        Context:
        {context}

        Question: {query}
        """
        response = ask_ollama(final_prompt, model=model, ollama_path=ollama_path)
        return response

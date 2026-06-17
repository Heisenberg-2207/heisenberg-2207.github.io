# NLP Suite

> CS6370 — Natural Language Processing · IIT Madras · Jan–May 2025  
> Plus two independent projects built alongside the course.

Four projects covering the NLP stack from classical retrieval through transformer fine-tuning to retrieval-augmented generation.

---

## Contents

```
NLP_Suite/
├── IR_System/               ← Modular hybrid information retrieval
├── DistilBERT_MentalHealth/ ← Transformer fine-tuning for mental health classification
├── RAG_Chatbot/             ← Offline RAG system with FAISS + local LLM
└── Gemini_Sentiment/        ← Prompt-engineered sentiment classification
```

---

## IR System — [`IR_System/`](./IR_System/) · Team of 5

**Task:** Given a query, retrieve the most relevant documents from the Cranfield aerodynamics corpus (1,400 docs, 225 queries).

**Pipeline:**

```
Raw query
    → Tokenization + Sentence Segmentation
    → Stopword Removal
    → Stemming / Lemmatization
    → Query Expansion (WordNet + Lesk WSD)
    → Spell Correction (edit distance)
    → Retrieval (TF-IDF / LSA / Autoencoder-LSA / SBERT)
    → Ranked results
```

**Results on Cranfield:**

| Model | NDCG@10 | MAP@10 | P@10 |
|---|---|---|---|
| TF-IDF | 0.41 | 0.38 | — |
| LSA (k=100) | 0.47 | 0.44 | — |
| Autoencoder-LSA | 0.49 | 0.46 | — |
| **SBERT** | **0.533** | **0.509** | best |

SBERT outperformed all baselines. Lesk-based WSD improved retrieval precision for ambiguous technical queries.

---

## DistilBERT Mental Health Classification — [`DistilBERT_MentalHealth/`](./DistilBERT_MentalHealth/)

**Task:** Classify Reddit/forum posts into 7 mental health emotional states (e.g., depression, anxiety, suicidal ideation, normal).

**Approach:**
- Fine-tuned DistilBERT on combined forum dataset
- Addressed class imbalance via weighted loss and oversampling
- Evaluation: F1-macro, per-class precision/recall

**Why DistilBERT:** 40% smaller than BERT-base, 60% faster, retains 97% of performance — practical for resource-constrained deployment in health settings.

---

## RAG PDF Chatbot — [`RAG_Chatbot/`](./RAG_Chatbot/)

A fully offline, local RAG system for querying multiple PDFs simultaneously.

**Architecture:**
```
PDF → chunks (token-based, with overlap)
    → MiniLM embeddings (sentence-transformers)
    → FAISS index (cosine similarity)
    → Top-k retrieval
    → Local LLM (Mistral / Llama2 via Ollama)
    → Answer with document + page citations
```

**Features:**
- Persistent embedding cache (avoid recomputation)
- Multi-document support
- Page-level citations in every answer
- Fully offline — no OpenAI API needed

**Usage:**
```python
from rag_system import RAGSystem
rag = RAGSystem(folder_path="documents/")
print(rag.answer_query("What did Aniket publish in 2026?"))
```

---

## Gemini Sentiment — [`Gemini_Sentiment/`](./Gemini_Sentiment/)

**Task:** Sentiment classification on Amazon Alexa product reviews.

**Approach:**
- Zero-shot and few-shot prompt engineering with Google Gemini
- Prompt templates designed for nuanced sentiment (positive / negative / mixed / irrelevant)
- Compared against fine-tuned DistilBERT as a baseline
- Integrated output into the mental health classifier pipeline for contextual analysis

**Takeaway:** Well-designed prompts with 3-shot examples matched fine-tuned DistilBERT accuracy without any gradient updates.

---

## Technologies

`Python 3.10+` · `PyTorch` · `Hugging Face Transformers` · `SBERT / sentence-transformers`  
`FAISS` · `Ollama` · `scikit-learn` · `NLTK` · `spaCy` · `Google Gemini API`

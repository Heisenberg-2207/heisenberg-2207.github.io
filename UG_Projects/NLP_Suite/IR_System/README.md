# Modular Information Retrieval System

> CS6370 — Natural Language Processing · IIT Madras · Jan–May 2025 · Team of 5

Hybrid information retrieval system evaluated on the Cranfield aerodynamics corpus, combining classical and neural methods with robust preprocessing.

## Dataset
**Cranfield corpus:** 1,400 aerodynamics abstracts, 225 queries, 1,837 relevance judgments.  
A classic IR benchmark from the 1960s, still widely used for evaluation.

## What's Here

```
IR_System/
├── IR system/
│   ├── main.py                 ← Entry point
│   ├── informationRetrieval.py ← Retrieval models (TF-IDF, LSA, SBERT)
│   ├── evaluation.py           ← Precision, Recall, MAP, NDCG
│   ├── tokenization.py         ← Tokenizer
│   ├── stopwordRemoval.py      ← Stopword filtering
│   ├── inflectionReduction.py  ← Stemming/lemmatization
│   ├── query_expansion.py      ← WordNet + Lesk WSD expansion
│   ├── spellcheck.py           ← Edit-distance spell correction
│   ├── word2vec.py             ← Word2Vec embeddings
│   ├── word2vec.npy            ← Pretrained Word2Vec vectors
│   ├── llm based/              ← SBERT and cross-encoder variants
│   ├── ablation study/         ← Dimension vs. accuracy analysis
│   ├── cranfield/              ← Dataset (docs, queries, qrels)
│   └── output/                 ← Preprocessed document cache
├── CS6370_Report.pdf           ← Full report
└── NLP Project_Short.pdf       ← Short summary
```

## Retrieval Models

| Model | Key Idea |
|---|---|
| **TF-IDF** | Term frequency weighted by inverse document frequency |
| **LSA** | SVD-based latent semantic space (dimension k) |
| **Autoencoder-LSA** | Nonlinear dimensionality reduction + LSA |
| **SBERT** | Sentence-BERT semantic embeddings with cosine similarity |

## Preprocessing Pipeline

1. Sentence segmentation
2. Tokenization (custom + NLTK)
3. Stopword removal
4. Stemming + lemmatization (Porter + WordNet)
5. Spell correction (edit distance / BK-tree)
6. Query expansion: WordNet synonyms + Improved Lesk WSD

## Results

| Model | NDCG@10 | MAP@10 |
|---|---|---|
| TF-IDF | 0.41 | 0.38 |
| LSA (k=100) | 0.47 | 0.44 |
| Autoencoder-LSA | 0.49 | 0.46 |
| **SBERT** | **0.533** | **0.509** |

## How to Run

```bash
cd "IR system"
python main.py --query "transonic flow over airfoils" --model sbert --top_k 10
```

Dependencies:
```bash
pip install nltk sentence-transformers scikit-learn numpy
```

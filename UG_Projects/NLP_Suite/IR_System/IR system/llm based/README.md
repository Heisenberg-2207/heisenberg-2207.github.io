# LLM-Based Retrieval Variants

Neural retrieval baselines layered on top of the core IR system, used to benchmark against the classical TF-IDF/LSA/Autoencoder-LSA pipeline.

| File | Description |
|---|---|
| `main.py` | `CrossEncoderInformationRetriever` + `LLMIRSystem` — re-ranks documents with a `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder after the standard preprocessing pipeline (tokenization, spell correction, stopword removal, lemmatization). |
| `sota.py` | SBERT-based retrieval using `all-MiniLM-L6-v2` sentence embeddings and cosine similarity — the best-performing model reported in the parent README (NDCG@10 = 0.533). |
| `evaluation.py` | Shared evaluation utilities (Precision/Recall/F-score/MAP/nDCG@k) used by `main.py`. |
| `util.py` | Shared helper functions. |
| `eval_plot.png` | Evaluation metric curves (Precision/Recall/F-score/MAP/nDCG vs. k) for the cross-encoder pipeline. |
| `eval_plot_cross_encoder.png` | Evaluation metric curves specifically for the cross-encoder re-ranking variant. |

Run directly with `python main.py` or `python sota.py` from this folder (expects `cran_docs.json`, `cran_queries.json`, `cran_qrels.json` from [`../cranfield/`](../cranfield/README.txt)).

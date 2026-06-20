# Output

Auto-generated intermediate cache from the preprocessing pipeline in `main.py` (see [`../README.txt`](../README.txt)). Each stage of preprocessing writes its output here for both documents and queries, so later stages don't need to recompute earlier ones.

| File | Pipeline Stage |
|---|---|
| `segmented_docs.txt` / `segmented_queries.txt` | After sentence segmentation |
| `tokenized_docs.txt` / `tokenized_queries.txt` | After tokenization |
| `stopword_removed_docs.txt` / `stopword_removed_queries.txt` | After stopword removal |
| `reduced_docs.txt` / `reduced_queries.txt` | After stemming/lemmatization (inflection reduction) |
| `eval_plot.png` | Precision/Recall/F-score/MAP/nDCG vs. k for the active retrieval model |

These are regenerated automatically when `main.py` is run; they don't need to be edited by hand.

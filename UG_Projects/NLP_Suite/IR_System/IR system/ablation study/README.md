# Ablation Study

Dimensionality ablation for the LSA and Autoencoder-LSA retrieval models — how retrieval quality varies with the latent dimension `k`.

| File | Description |
|---|---|
| `dimension_vs_accuracy_study.ipynb` | Sweeps `k` (latent dimension) for LSA and Autoencoder-LSA and records Precision/Recall/F-Score/MAP/nDCG@10 at each value |
| `Precision\|Recall\|F-Score\|MAP\|nDCG Autoencoder score vs K for different dimensions.png` | Per-metric ablation curves for the Autoencoder-LSA model |
| `Precision\|Recall\|F-Score\|MAP\|nDCG LSA score vs K for different dimensions.png` | Per-metric ablation curves for the plain LSA model |
| `singular_values_plot.png` | Singular value spectrum of the TF-IDF term-document matrix (motivates the choice of `k`) |
| `variance_captured_plot.png` | Cumulative variance explained vs. number of LSA components |
| `base model ir.png` | Baseline (non-ablated) model performance for reference |

These plots support the model-selection discussion in the parent [`IR_System/README.md`](../../README.md) (SBERT outperforming TF-IDF/LSA/Autoencoder-LSA at their respective best `k`).

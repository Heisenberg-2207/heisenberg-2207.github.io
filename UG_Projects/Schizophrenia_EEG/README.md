# Graph-Based Classification of Schizophrenia from Resting-State EEG

> Computational Neuroscience Lab, IIT Madras · Mount Sinai Hospital collaboration · May–Jul 2024

A graph convolutional network (GCNN) pipeline that classifies schizophrenic (CSZ) vs. healthy (HE) individuals from resting-state EEG, using power-spectral and inter-electrode correlation features as graph node/edge inputs.

## Dataset

EEG recordings (`.set` / EEGLAB format) for **23 CSZ and 40 HE subjects**, provided through the Mount Sinai collaboration, under two recording conditions:
- **EC** — eyes closed
- **EO** — eyes open

## Pipeline

```
Raw EEG (.set, 512 Hz)
    → Butterworth band-pass filtering (delta 0.1–4 Hz / theta 4–7 Hz / alpha 8–12 Hz)
    → Electrode clustering (cluster sizes: 2, 5, 8, 12)
    → Power Spectral Density (Welch's method) per cluster  → node features
    → Pearson correlation matrix across clusters            → adjacency/edge input
    → Graph Convolutional Network (Keras GCN) classifier
    → CSZ / HE prediction
```

Each subject's EEG is reduced to a small graph: nodes are electrode clusters carrying PSD-based feature vectors, and edges are weighted by inter-cluster Pearson correlation. The GCN passes messages over this graph (`GraphConv`, 3-step propagation) before a dense softmax head predicts the binary class label. Models were evaluated with a round-robin (leave-one-out style) train/test split across the 23 CSZ / 40 HE subjects, separately for each frequency band, cluster size, and EC/EO condition.

## What's Here

```
Schizophrenia_EEG/
└── scripts/
    ├── make_batches.ipynb / make_batches_ver2.ipynb   ← Load raw .set files, filter, segment into batches
    ├── power_spectrum.ipynb                            ← Welch PSD computation per band/cluster
    ├── corr_study.ipynb / corr_study_ver2.ipynb        ← Pearson correlation matrix construction
    ├── pearson_mount_sinai.ipynb / _ver2.ipynb         ← GCN model definition, training, and evaluation
    ├── label.ipynb                                     ← Label assignment for CSZ/HE batches
    ├── supreme.ipynb                                   ← Consolidated pipeline: loops over all bands × clusters, writes results table
    ├── ANOVA.ipynb                                      ← Statistical (ANOVA) analysis of classification accuracy across conditions
    ├── table_EC.csv                                     ← Per-band/cluster classification results, eyes-closed
    └── table_EO.csv                                     ← Per-band/cluster classification results, eyes-open
```

## Model

A two-input Keras model (`keras_gcn.GraphConv`):
- **Node input:** PSD feature vectors per electrode cluster (shape: `num_clusters × node_dim`)
- **Edge input:** Pearson correlation adjacency matrix (shape: `num_clusters × num_clusters`)
- **Architecture:** `GraphConv(300, steps=3) → GraphConv(1, steps=3) → Dense(50, tanh) → Dense(2, softmax)`
- **Training:** Adam (lr=1e-3), binary cross-entropy loss, 50 epochs

## Results

Classification accuracy and statistical significance (via ANOVA) were compared across frequency bands (delta/theta/alpha), electrode cluster granularities, and eyes-open vs. eyes-closed conditions to identify which combination best separates CSZ from HE subjects — recorded in `table_EC.csv` and `table_EO.csv`.

This project was the entry point into the neuroscience research that subsequently led to the RLC 2026 and AD/PD 2026 publications.

## Technologies

`Python` · `MNE` (EEG processing) · `SciPy` (filtering, Welch PSD) · `TensorFlow / Keras` · `keras-gcn` (GraphConv) · `scikit-learn` · `statsmodels` (ANOVA) · `pandas` · `NumPy`

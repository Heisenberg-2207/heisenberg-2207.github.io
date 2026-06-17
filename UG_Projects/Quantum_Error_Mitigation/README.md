# Quantum Error Mitigation with Machine Learning

> DA6300 — Quantum Machine Learning · IIT Madras · Jan–May 2025 · Team of 2

Explored ML-based alternatives to Zero-Noise Extrapolation (ZNE) for mitigating noise in quantum circuits, tested on IBM Q Brisbane (127-qubit hardware).

## Background

Quantum computers are noisy. Every gate introduces errors; deeper circuits accumulate more. **Zero-Noise Extrapolation (ZNE)** is the current gold standard: run the circuit at multiple noise levels and extrapolate to zero noise. It works but is computationally expensive (multiple circuit executions per mitigation).

This project asks: **can a trained ML model mitigate noise with a single circuit execution?**

## What's Here

```
Quantum_Error_Mitigation/
├── AQUA/                       ← Main simulator-based pipeline
│   ├── main.ipynb              ← Primary experiment notebook
│   ├── converter.py            ← Circuit → graph/frequency/CNN encoding
│   ├── extractor.py            ← Feature extraction
│   ├── generator.py            ← Circuit generation
│   ├── simulator.py            ← Noisy simulation
│   └── zne.py                  ← ZNE baseline implementation
├── hardware/                   ← IBM Q Brisbane experiments
│   ├── main.ipynb              ← Hardware experiment notebook
│   ├── gnn_predictor.ipynb     ← GNN model on hardware data
│   └── ibm_brisbane_calibrations_*.csv
├── pipeline/src/
│   ├── DATA_GEN/               ← Synthetic data generation
│   └── QEM/                    ← ML models (GCN, GAT, KAN)
├── results/                    ← Figures (ANN, CNN, KAN, XGB, RF results)
└── demo2.pdf                   ← Full project report
```

## Encodings

| Encoding | Method | ML Model |
|---|---|---|
| **Graph** | Circuit topology as DAG | GCN, GAT |
| **Frequency** | Fourier spectrum of output distribution | CNN, RNN |
| **Matrix** | Output bitstring matrix | ANN, XGBoost |

## Models

GCN · GAT · KAN · ANN · CNN · RNN · XGBoost · Random Forest

## Results

Best ML models achieved **up to 9× accuracy improvement** over ZNE on shallow circuits, while eliminating the runtime overhead of multiple circuit executions.

KAN achieved the lowest L2 error and MAE across all tested models.

## Hardware Validation

Experiments were run on **IBM Q Brisbane** (127-qubit). Calibration data included. ML models trained on simulator data showed partial generalization to hardware noise profiles.

## Technologies

`Qiskit` · `PyTorch` · `PyTorch Geometric` · `Scikit-learn` · `XGBoost` · `NumPy`

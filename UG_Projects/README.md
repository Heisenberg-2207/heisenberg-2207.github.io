# Undergraduate Projects

> B.Tech Mechanical Engineering + Minor in AI/ML — IIT Madras (2021–2025)  
> CGPA: 8.93/10 · Minor CGPA: 9.5/10

This folder contains all major projects completed during my undergraduate studies at IIT Madras. They span reinforcement learning, natural language processing, computer vision, quantum computing, and high-performance computing — reflecting a deliberately broad technical foundation built toward a research career in RL and computational neuroscience.

---

## Projects at a Glance

| Folder | Project | Course / Lab | Stack |
|---|---|---|---|
| [`RL_Suite/`](./RL_Suite/README.md) | Foundations of RL: Q-Learning → Options | CS6700 (Reinforcement Learning) | PyTorch, Gymnasium |
| [`NLP_Suite/`](./NLP_Suite/README.md) | Four NLP projects across IR, classification, and generation | CS6370 (NLP) + Independent | BERT, FAISS, SBERT |
| [`Localisation_and_Mapping/`](./Localisation_and_Mapping/README.md) | YOLOv8 wheelchair detection & docking | BioNeX Lab UG Research | YOLOv8, OpenCV |
| [`Quantum_Error_Mitigation/`](./Quantum_Error_Mitigation/README.md) | ML-based QEM on IBM quantum hardware | DA6300 (Quantum ML) | Qiskit, PyTorch Geometric |
| [`Parallel_DFT/`](./Parallel_DFT/README.md) | 2D DFT from scratch, parallelized | ID5130 (Parallel Computing) | C/C++, OpenMP, MPI |
| [`Schizophrenia_EEG/`](./Schizophrenia_EEG/README.md) | GCNN on resting-state EEG for schizophrenia classification | Comp. Neuroscience Lab (Mount Sinai collab) | PyG, PyTorch |

---

## RL Suite — [`RL_Suite/`](./RL_Suite/README.md)

Three progressive assignments covering foundational through advanced RL algorithms.

- **Q-Learning & SARSA** — Tabular methods, Bayesian hyperparameter search, custom GridWorld
- **DDQN & REINFORCE** — Deep value-based and policy-gradient methods on CartPole and Acrobot
- **SMDP & Intra-Option RL** — Hierarchical RL with options, pseudo-reward analysis, Taxi environment

Collectively these form a structured survey of the RL landscape from tabular methods to options frameworks.

---

## NLP Suite — [`NLP_Suite/`](./NLP_Suite/README.md)

Four projects across the NLP stack:

- **IR System** — Modular hybrid retrieval (TF-IDF, LSA, Autoencoder, SBERT) on Cranfield; NDCG@10 = 0.533
- **DistilBERT Mental Health** — Fine-tuned DistilBERT for 7-class emotional state classification
- **RAG Chatbot** — Offline PDF question-answering with FAISS + local LLM (Ollama)
- **Gemini Sentiment** — Prompt-engineered Gemini pipeline for Amazon Alexa review sentiment

---

## Localisation and Mapping — [`Localisation_and_Mapping/`](./Localisation_and_Mapping/README.md)

Computer vision pipeline for wheelchair localization and automatic docking, developed during the **BioNeX Lab Undergraduate Research Project** under Dr. Manish Anand (Dept. of Mechanical Engineering, IIT Madras).

Trained a YOLOv8 model on a custom dataset of 500+ annotated wheelchair images. Outputs position estimates within **±2 cm** uncertainty, integrated into a semi-autonomous wheelchair docking pipeline.

---

## Quantum Error Mitigation — [`Quantum_Error_Mitigation/`](./Quantum_Error_Mitigation/README.md)

Explored ML-based alternatives to Zero-Noise Extrapolation (ZNE) for quantum circuit noise mitigation. Implemented GNNs, GATs, KANs, and classical models (XGBoost, Random Forest), tested on both simulators and **IBM Q Brisbane** (127-qubit hardware). Achieved up to **9× accuracy improvement** over ZNE on shallow circuits.

---

## Parallel DFT — [`Parallel_DFT/`](./Parallel_DFT/README.md)

Implemented the 2D Discrete Fourier Transform from scratch in C/C++ — no FFTW, no numpy. Parallelized with **OpenMP** (shared memory) and **MPI** (distributed), achieving over **80% efficiency** with near-linear speedup at 1:1 thread-to-speedup correspondence.

---

## Schizophrenia EEG Classification — [`Schizophrenia_EEG/`](./Schizophrenia_EEG/README.md)

GCNN pipeline for automated schizophrenia detection from resting-state EEG, developed during the Computational Neuroscience Lab internship in collaboration with **Mount Sinai Hospital**. Built functional connectivity graphs from PSD features (delta/theta/alpha bands) and evaluated under eyes-open and eyes-closed conditions using leave-one-out cross-validation. This project was the entry point into the neuroscience research that led to the AD/PD and RLC publications.

---

## Technical Stack (across all projects)

**Languages:** Python, C, C++  
**ML/DL:** PyTorch, Scikit-learn, PyTorch Geometric, Hugging Face Transformers  
**RL:** Gymnasium, Stable-Baselines3, custom environments  
**NLP:** BERT, DistilBERT, SBERT, Word2Vec, TF-IDF  
**Vision:** YOLOv8, OpenCV  
**Quantum:** Qiskit, IBM Q hardware  
**HPC:** OpenMP, MPI  
**Tools:** FAISS, Weights & Biases, Ollama, Matplotlib, NumPy

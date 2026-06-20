# Aniket Khan — Research & Project Portfolio

> Incoming PhD Candidate in Computer Science @ NUS · B.Tech + Minor in AI/ML, IIT Madras · 8.93 CGPA

My work sits at the intersection of **reinforcement learning**, **computational neuroscience**, and **machine learning**. I am interested in understanding decision-making in both biological agents (the brain, the gut-brain axis) and artificial ones, and in building principled computational tools for scientific discovery.

---

## Structure

```
portfolio/
│
├── NutriRL/                      ← RLC 2026 publication (benchmark)
├── ADPD_Appetite_Regulation/     ← AD/PD 2026 publication (poster)
│
└── UG_Projects/                  ← Undergraduate coursework & independent projects
    ├── RL_Suite/                 ← Three RL assignments (Q-Learning → DDQN → Options)
    ├── NLP_Suite/                ← Four NLP projects (IR, BERT, RAG, Gemini)
    ├── Localisation_and_Mapping/ ← YOLOv8 wheelchair detection (BioNeX Lab)
    ├── Quantum_Error_Mitigation/ ← GNN/KAN-based QEM on IBM Q hardware
    ├── Schizophrenia_EEG/        ← GCNN classification on resting-state EEG (Mount Sinai collab)
    └── Parallel_DFT/             ← 2D DFT via OpenMP + MPI from scratch
```

---

## Publications

| Paper | Venue | Status |
|---|---|---|
| NutriRL: A Benchmark for Nutritional Regulation under Delayed State Transitions | RLC 2026 / RLJ | Accepted |
| RL Model of Dopamine-Mediated Appetite Regulation (Healthy, Parkinson's, Medication) | AD/PD 2026 | Accepted (Poster) |

---

## Research Projects (Published / Ongoing)

### [`NutriRL/`](./NutriRL/README.md)
A Gym-compatible RL benchmark for food-choice decision making under **delayed physiological feedback**. Evaluates PPO, DDQN, SAC, and MC (REINFORCE-style) agents under controlled delay regimes, with a GRU-based recurrent policy variant (GRU-PPO) also implemented. Models nutrient absorption with Gaussian-kernel dynamics and irreversible accumulation. *RLC 2026, to appear in RLJ.*

### [`ADPD_Appetite_Regulation/`](./ADPD_Appetite_Regulation/README.md)
A cortico-basal ganglia RL model of appetite regulation. Dopamine is mapped to TD-error signals; the model reproduces clinical dietary patterns across healthy, Parkinsonian, and medicated states. *AD/PD 2026 poster.*

---

## Undergraduate Projects

See [`UG_Projects/README.md`](./UG_Projects/README.md) for the full overview.

Quick links:

| Project | Key Technologies | What it does |
|---|---|---|
| [RL Suite](./UG_Projects/RL_Suite/README.md) | PyTorch, Gymnasium | Q-Learning → DDQN → Options, full suite |
| [NLP Suite](./UG_Projects/NLP_Suite/README.md) | BERT, FAISS, SBERT | IR, mental-health classification, RAG chatbot |
| [Localisation & Mapping](./UG_Projects/Localisation_and_Mapping/README.md) | YOLOv8, OpenCV | Wheelchair detection & docking (±2 cm) |
| [Quantum Error Mitigation](./UG_Projects/Quantum_Error_Mitigation/README.md) | Qiskit, PyG | GNN/KAN vs ZNE on IBM Brisbane |
| [Schizophrenia EEG](./UG_Projects/Schizophrenia_EEG/README.md) | Keras GCN, MNE | GCNN classification on resting-state EEG (Mount Sinai collab) |
| [Parallel DFT](./UG_Projects/Parallel_DFT/README.md) | C/C++, OpenMP, MPI | 2D DFT from scratch, >80 % efficiency |

---

## Contact

**Email:** aniketkhan2003@gmail.com  
**GitHub:** [Heisenberg-2207](https://github.com/Heisenberg-2207)  
**LinkedIn:** [aniket-khan-790855226](https://www.linkedin.com/in/aniket-khan-790855226)

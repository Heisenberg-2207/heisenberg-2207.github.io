# RL Model of Dopamine-Mediated Appetite Regulation

> **AD/PD 2026** — Accepted for Poster Presentation  
> International Conference on Alzheimer's and Parkinson's Diseases  
> Joint first authorship with Charitha Palika

---

## Overview

This project develops a **cortico-basal ganglia reinforcement learning model** of appetite regulation and food-choice behavior, simulating how dopamine dysfunction in Parkinson's Disease — and its pharmacological treatment — alters eating behavior.

The core insight: **dopamine release is computationally equivalent to a temporal-difference (TD) error signal**. By manipulating the dopamine dynamics in the RL model, we can reproduce clinically observed dietary patterns across three conditions.

---

## Model Architecture

### Dopamine → TD Error Mapping
Dopamine phasic activity is modeled as the TD prediction error:
```
δ(t) = r(t) + γ·V(s_{t+1}) − V(s_t)
```
Where `δ(t)` drives both value learning (striatal) and policy updates (cortical).

### Three Simulated Conditions

| Condition | Dopamine Model | Predicted Behavior |
|---|---|---|
| **Healthy** | Normal DA release (phasic + tonic) | Caloric balance, regular meals |
| **Parkinsonian** | Reduced DA (degeneration model) | Caloric deficit, reduced food drive |
| **Dopaminergic Treatment** | Supraphysiological phasic DA | Compulsive overeating, impulsive choices |

All three reproduce **experimentally observed clinical dietary intake patterns**, validating the biological plausibility of the computational approach.

---

## Key Results

- Model successfully reproduces the **hedonic hyperphagia** seen in dopamine-medicated Parkinson's patients
- TD error timing and magnitude predict meal initiation and meal size
- Demonstrates that altered **reward prediction** — not appetite per se — drives the behavioral changes
- Provides a mechanistic account of dopamine agonist side effects (impulse control disorders)

---

## Files

```
ADPD_Appetite_Regulation/
└── ADPD_e_poster.pptx.pdf    ← Conference e-poster (AD/PD 2026)
```

---

## Clinical Relevance

This work contributes to understanding why Parkinson's patients on dopamine agonists (e.g., pramipexole, ropinirole) develop compulsive behaviors including binge eating. The RL framework offers a principled computational account that can inform dosing strategies and behavioral interventions.

---

## Citation

```bibtex
@inproceedings{khan2026dopamine,
  title     = {A Reinforcement Learning Model of Dopamine-Mediated Appetite Regulation
               under Healthy, Parkinsonian, and Dopaminergic Treatment Conditions},
  author    = {Khan, Aniket and Palika, Charitha},
  booktitle = {International Conference on Alzheimer's and Parkinson's Diseases (AD/PD)},
  year      = {2026},
  note      = {Poster presentation}
}
```

---

## Technologies

`Python` · `PyTorch` · `NumPy` · `TD Learning` · `DDPG` · `Matplotlib`

---

## Authors

- **Aniket Khan** · Computational Neuroscience Lab, IIT Madras → NUS PhD
- **Charitha Palika** · IIT Madras

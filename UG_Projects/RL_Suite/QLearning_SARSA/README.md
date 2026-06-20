# Q-Learning & SARSA on Custom GridWorld

> CS6700 Assignment 1 · IIT Madras · Jan 2024

Tabular RL methods with Bayesian hyperparameter optimization and exploration strategy comparison on a custom-built stochastic GridWorld.

## What's Here

| File | Description |
|---|---|
| `PA1.ipynb` | Full experiment notebook |
| `GridWorld_PA1.ipynb` | GridWorld environment definition |
| `gridworld_pa1.py` | Python module for the environment |
| `Report.pdf` | Full write-up with results |
| `world_img.png` | GridWorld visualization |

## Algorithm Summary

**Q-Learning** (off-policy):  
`Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') − Q(s,a)]`

**SARSA** (on-policy):  
`Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') − Q(s,a)]`

The key difference: Q-Learning learns the greedy policy regardless of what it actually does; SARSA learns the policy it follows. This makes SARSA safer near dangerous states.

## Key Results

- SARSA was more conservative near cliff states, avoiding high-risk paths
- Q-Learning found the optimal policy ~15% faster under deterministic transitions
- Softmax exploration outperformed ε-greedy when reward differences were small
- Bayesian search identified optimal `α ≈ 0.3`, `γ ≈ 0.95` for this environment

> Full derivations, hyperparameter tables, and extended results are documented in `Report.pdf`.

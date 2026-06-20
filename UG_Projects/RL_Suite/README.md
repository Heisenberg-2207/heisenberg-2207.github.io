# Reinforcement Learning Suite

> CS6700 — Reinforcement Learning · IIT Madras · Jan–May 2024

A structured progression through foundational and advanced RL algorithms, covering tabular methods, deep RL, and hierarchical control. Three assignments, each building on the last.

---

## Contents

```
RL_Suite/
├── QLearning_SARSA/      ← Assignment 1: Tabular RL
├── DDQN_REINFORCE/       ← Assignment 2: Deep RL
└── SMDP_IntraOption/     ← Assignment 3: Hierarchical RL
```

---

## Assignment 1 — Q-Learning & SARSA [`QLearning_SARSA/`](./QLearning_SARSA/)

**Environment:** Custom-built 5×5 GridWorld with configurable rewards, obstacles, and stochastic transitions.

**Algorithms implemented:**
- Q-Learning (off-policy TD)
- SARSA (on-policy TD)

**Experiments:**
- Epsilon-greedy vs. softmax (Boltzmann) exploration
- Bayesian hyperparameter optimization for learning rate and discount factor
- State visitation heatmaps and reward convergence curves
- Analysis of on-policy vs. off-policy divergence under noise

**Key result:** SARSA was more conservative near "cliff" states due to on-policy evaluation; Q-Learning converged faster to the optimal policy under deterministic transitions.

---

## Assignment 2 — DDQN & REINFORCE [`DDQN_REINFORCE/`](./DDQN_REINFORCE/)

**Environments:** `CartPole-v1`, `Acrobot-v1` (OpenAI Gymnasium)

**Algorithms implemented:**
- Dueling Deep Q-Network (DDQN) with experience replay
- REINFORCE (Monte Carlo policy gradient) — with and without baseline

**Key design choices:**
- Separate value and advantage streams in Dueling DQN architecture
- Variance reduction study: REINFORCE-with-baseline vs. without
- Learning curves averaged over multiple random seeds

**Key result:** Baseline subtraction reduced variance in REINFORCE by ~40%; Dueling DQN converged ~2× faster on Acrobot-v1.

---

## Assignment 3 — SMDP & Intra-Option RL [`SMDP_IntraOption/`](./SMDP_IntraOption/)

**Environment:** `Taxi-v3` (OpenAI Gymnasium) with custom options

**Algorithms implemented:**
- Semi-Markov Decision Processes (SMDP Q-Learning)
- Intra-Option Q-Learning (Sutton et al.)
- Fixed options (handcrafted sub-policies)
- Learnable options via pseudo-reward shaping

**Experiments:**
- Comparison of flat Q-Learning vs. SMDP vs. Intra-Option
- Analysis of option reuse and temporal abstraction
- Pseudo-reward function design and its effect on option quality

**Key result:** Intra-Option Q-Learning with pseudo-rewards converged to near-optimal policy in ~60% of the flat Q-Learning training steps by leveraging temporal abstraction.

---

## Technologies

`Python 3.10+` · `PyTorch` · `Gymnasium` · `NumPy` · `Matplotlib` · `Weights & Biases`

---

## How to Run

Each sub-folder contains a Jupyter notebook (`.ipynb`) that can be run top-to-bottom. Dependencies:

```bash
pip install gymnasium torch numpy matplotlib
```

For W&B logging (optional):
```bash
pip install wandb
wandb login
```

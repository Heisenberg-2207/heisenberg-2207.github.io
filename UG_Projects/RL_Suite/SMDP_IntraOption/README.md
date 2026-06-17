# SMDP & Intra-Option Reinforcement Learning

> CS6700 Assignment 3 · IIT Madras · Apr–May 2024

Hierarchical RL using Semi-Markov Decision Processes and the Options framework on Taxi-v3.

## Environment
**Taxi-v3** (OpenAI Gymnasium) — a discrete navigation task requiring the agent to pick up and drop off a passenger efficiently. Custom options (sub-policies) are defined over this environment.

## What's Here

| File | Description |
|---|---|
| `script.ipynb` | Full implementation notebook |
| `PA3.pdf` | Assignment report |
| `Result.png` | Training curve comparisons |

## Algorithms

### SMDP Q-Learning
Extends standard Q-Learning to handle variable-duration actions (options).  
Temporal discounting accounts for the number of primitive steps an option executes.

### Intra-Option Q-Learning (Sutton et al. 1999)
Updates option values using transitions that occur *within* an option's execution, not just at its termination. More sample-efficient than SMDP Q-Learning.

### Options
Each option is a tuple `(I, π, β)`:
- `I` — initiation set (where this option can start)
- `π` — internal policy (what to do while executing)
- `β` — termination condition (when to stop)

**Fixed options:** handcrafted sub-policies (e.g., "navigate to pickup location")  
**Learnable options:** internal policy learned via pseudo-reward shaping

## Key Results

- Intra-Option Q-Learning with pseudo-rewards converged in ~60% of the steps needed by flat Q-Learning
- Temporal abstraction allowed the agent to plan at multiple timescales simultaneously
- Pseudo-reward design was the critical factor: poorly shaped rewards led to premature option termination

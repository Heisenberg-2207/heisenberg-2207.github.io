# Dueling DQN & REINFORCE

> CS6700 Assignment 2 · IIT Madras · Feb–Mar 2024

Deep RL implementations covering value-based and policy-gradient methods on standard Gymnasium benchmarks.

## Environments
- **CartPole-v1** — balance a pole on a cart (discrete actions)
- **Acrobot-v1** — swing a two-link pendulum to a target height (discrete actions)

## What's Here

| File | Description |
|---|---|
| `Dueling_DQN.ipynb` | Full Dueling DQN implementation |
| `Reinforce_withBaseline.ipynb` | REINFORCE + learned baseline |
| `Reinforce_withoutBaseline.ipynb` | Vanilla REINFORCE |
| `PA2.pdf` | Assignment specification & report |
| `Result.png` | Training curves |

## Algorithms

### Dueling Deep Q-Network (DDQN)
Separates state-value V(s) and advantage A(s,a) in the network head:  
`Q(s,a) = V(s) + [A(s,a) − mean_a A(s,a)]`

Features: experience replay buffer, target network, epsilon-greedy decay.

### REINFORCE (Monte Carlo Policy Gradient)
`∇J(θ) = E[∇ log π_θ(a|s) · G_t]`

**With baseline:** subtracts a learned V(s) to reduce variance without introducing bias.  
**Without baseline:** higher variance but simpler to implement.

## Key Results

- Dueling DQN solved CartPole-v1 in ~200 episodes (baseline: ~350)
- REINFORCE-with-baseline reduced gradient variance by ~40% vs. without
- Acrobot: DDQN converged ~2× faster than REINFORCE across seeds

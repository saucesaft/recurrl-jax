# recurrl-jax

[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-0.7%2B-orange)](https://github.com/google/jax)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.3%2B-green)](https://github.com/google-deepmind/mujoco)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

A JAX-based recurrent reinforcement learning library. Supports LSTM, GRU, and GTrXL (Gated Transformer-XL) sequence models with PPO and A2C. Built for use with MuJoCo/MJX environments and designed around asymmetric actor-critic training.

## features

- **sequence models**: multi-layer LSTM, GRU, and GTrXL with correct episode boundary resets
- **agents**: PPO (with adaptive LR, minibatch BPTT) and A2C
- **asymmetric actor-critic**: policy obs and privileged obs split at the model level
- **domain randomization**: per-environment batched MJX model randomization
- **training**: configurable via Hydra, Orbax checkpointing, and video recording

## example: LEAP hand reorientation

Dexterous in-hand reorientation with a 16-DOF LEAP hand in MJX.

```bash
cd examples/leap_hand
uv run python train.py
```

Config lives in `examples/leap_hand/config/`.

## library structure

```
recurrl_jax/
├── agents/          # PPO, A2C
├── model_fns/       # factory functions for actor, critic, repr, seq models
├── models/
│   ├── actor_critic.py
│   ├── rnns/        # LSTM, GRU (multi-layer)
│   └── transformers/  # GTrXL
├── trainers/        # Trainer, BaseTrainer
└── utils/           # wrappers, logging, video, quat math, running stats
```

## acknowledgments

- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) — Google DeepMind. The LEAP hand mesh assets and XML scene descriptions are sourced from this repository.
- [BRAX](https://github.com/google/brax) — Freeman et al., Google DeepMind. The batched MJX domain randomization approach follows patterns established in Brax.
- [subho406](https://github.com/subho406) — their [Recurrent PPO with JAX](https://github.com/subho406/Recurrent-PPO-Jax) was a reference for the recurrent PPO implementation.
- [GTrXL](https://arxiv.org/abs/1910.06764) — Parisotto et al., "Stabilizing Transformers for Reinforcement Learning".
- [CleanRL](https://github.com/vwxyzjn/cleanrl) — reference for the PPO implementation structure.

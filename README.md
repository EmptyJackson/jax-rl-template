# jax-rl-template
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A minimal JAX-based reinforcement learning template, for rapidly spinning up RL projects!

All training and evaluation is JIT-compiled end-to-end in JAX. The template is for `Python 3.8.12`, built on top of:
* [JAX](https://github.com/google/jax) - Autograd and XLA
* [Flax](https://github.com/google/flax) - Neural network library
* [Optax](https://github.com/deepmind/optax) - Gradient-based optimisation
* [Distrax](https://github.com/deepmind/distrax) - Probability distributions
* [Weights & Biases](https://wandb.ai/site) - Experiment tracking and visualisation


## Features
Variants of this template are released as branches of this repository, each with different features:
| Branch | Description | Agents | Environments |
| --- | --- | --- | --- |
| `main` (here) | Basic training and evaluation functionality (e.g. training loop, logging, checkpointing), plus common online RL agents | `PPO`, `SAC`, `DQN` | [`Gymnax`](https://github.com/RobertTLange/gymnax) |
| `offline` (TBC) | Adds offline RL functionality (e.g. replay buffer, offline training) | `CQL`, `EDAC` | - |

This template is designed to provide only core functionality, providing a solid foundation for RL projects. Whilst it is not designed to be a full-featured RL library, please raise an issue if you think a feature is missing that would be useful for many projects.


## Setup

### Running locally (CPU)

1. **Install Python packages** from `requirements-base.txt` and `requirements-cpu.txt` in `setup` with:
```
cd setup && pip install $(cat requirements-base.txt requirements-cpu.txt)
```
2. **Sign into WandB** to enable logging:
```
wandb login
```

### Running via Docker

1. **Build the Docker container** with the provided script:
```
cd setup/docker && ./build.sh
```
2. **Add your [WandB key](https://wandb.ai/authorize)** to the `setup/docker` folder:
```
echo <wandb_key> > setup/docker/wandb_key
```

### Automatic code formatting
**Install the [Black](https://github.com/psf/black) pre-commit hook**, after installing Python packages, with:
```
pre-commit install
```
This will check and fix formatting errors when you commit code.

## Usage

### Training locally

To train an agent, run:
```
python train.py <arguments>
```
For example, to train a PPO agent on the CartPole-v1 environment and log to WandB, run:
```
python train.py --agent ppo --env CartPole-v1 --log --wandb_entity wandb_username --wandb_project project_name
```
To see all possible arguments, see `experiments/parse_args.py` or run:
```
python train.py --help
```

### Training via Docker
Launch training runs inside your built container with:
```
./run_docker.sh <gpu_id> train.py <arguments>
```
For example, to train a DQN agent on the Asterix-MinAtar environment using GPU 3, run:
```
./run_docker.sh 3 train.py --agent dqn --env Asterix-MinAtar
```

## Acknowledgements
Large parts of the training loop and PPO implementation are based on [PureJaxRL](https://github.com/luchris429/purejaxrl), which contains high-performance, single-file implementations of RL agents in JAX.

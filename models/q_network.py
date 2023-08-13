import distrax
import jax.numpy as jnp
import flax.linen as nn

from typing import Sequence
from flax.linen.initializers import constant, orthogonal


class QNetwork(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        critic = nn.Dense(
            64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            self.action_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(critic)

        # Use epsilon-greedy policy
        eps = 0.1
        probs = (
            jnp.ones_like(critic) * eps / self.action_dim
            + (1 - eps) * jnp.eye(self.action_dim)[jnp.argmax(critic, axis=-1)]
        )
        pi = distrax.Categorical(probs=probs)

        return pi, critic

    def init_args(self, obs_shape, num_actions):
        return (jnp.zeros(obs_shape),)

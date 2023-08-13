import distrax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SoftQNetwork(nn.Module):
    activation: str = "tanh"

    @nn.compact
    def __call__(self, obs, action):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        x = jnp.concatenate([obs, action], axis=-1)
        x = nn.Dense(64)(x)
        x = activation(x)
        x = nn.Dense(64)(x)
        x = activation(x)
        q = nn.Dense(1)(x)
        return jnp.squeeze(q, axis=-1)

    def init_args(self, obs_shape, num_actions):
        return (jnp.zeros(obs_shape), jnp.zeros(num_actions))


class VectorCritic(nn.Module):
    activation: str = "tanh"
    n_critics: int = 2

    @nn.compact
    def __call__(self, obs, action):
        # Idea taken from https://github.com/perrin-isir/xpag
        # Similar to https://github.com/tinkoff-ai/CORL for PyTorch
        vmap_critic = nn.vmap(
            SoftQNetwork,
            variable_axes={"params": 0},  # parameters not shared between the critics
            split_rngs={"params": True, "dropout": True},  # different initializations
            in_axes=None,
            out_axes=-1,
            axis_size=self.n_critics,
        )
        q_values = vmap_critic(activation=self.activation)(obs, action)
        return q_values

    def init_args(self, obs_shape, num_actions):
        return (jnp.zeros(obs_shape), jnp.zeros(num_actions))


class TanhGaussianActor(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    action_lims: Sequence[float] = (-1.0, 1.0)

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_x = nn.Dense(64)(x)
        actor_x = activation(actor_x)
        actor_x = nn.Dense(64)(actor_x)
        actor_x = activation(actor_x)
        actor_mean = nn.Dense(self.action_dim)(actor_x)
        actor_logstd = nn.Dense(self.action_dim)(actor_x)
        actor_logstd = jnp.clip(actor_logstd, LOG_STD_MIN, LOG_STD_MAX)
        action_scale = (self.action_lims[1] - self.action_lims[0]) / 2
        action_bias = (self.action_lims[1] + self.action_lims[0]) / 2
        pi = distrax.Transformed(
            # distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logstd)),
            distrax.Normal(actor_mean, jnp.exp(actor_logstd)),
            # Note: Chained bijectors applied in reverse order
            distrax.Chain(
                [
                    distrax.Lambda(lambda x: (x * action_scale) + action_bias),
                    distrax.Tanh(),
                ]
            ),
        )
        # No value function
        return pi, None

    def init_args(self, obs_shape, num_actions):
        return (jnp.zeros(obs_shape),)


class EntropyCoef(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_ent_coef = self.param(
            "log_ent_coef",
            init_fn=lambda key: jnp.full((), jnp.log(self.ent_coef_init)),
        )
        return jnp.exp(log_ent_coef)

    def init_args(self, obs_shape, num_actions):
        # No input
        return tuple([])

import jax
import gymnax
import jax.numpy as jnp

from typing import Optional
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper, GymnaxWrapper
from gymnax.environments import spaces

from util import Transition


def get_env(env_name: str, env_kwargs: dict):
    if env_name in gymnax.registered_envs:
        env, env_params = gymnax.make(env_name, **env_kwargs)
    else:
        raise ValueError(
            f"Environment {env_name} not registered in any environment sources."
        )
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)
    return env, env_params


class ClipAction(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, key, state, action, params=None):
        action = jnp.clip(
            action,
            self._env.action_space(params).low,
            self._env.action_space(params).high,
        )
        return self._env.step(key, state, action, params)


class RolloutWrapper:
    def __init__(
        self,
        env_name: str,
        num_env_steps: Optional[int] = None,
        env_kwargs: dict = {},
    ):
        """
        Wrapper providing batch agent rollout.

        Assumes the same environment kwargs (same state and action space) but different environment parameters.

        E.g. a collection of tabular agents with different weights, over CartPole environments
        with difference environment settings per agent.

        Args:
            env_name (str): Name of environment to use.
            num_env_steps (int): Number of environment steps to run per agent.
            env_kwargs (dict): Static keyword arguments to pass to environment, same for all agents.
        """
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        # Define the environment & network forward function
        self.env, self.default_env_params = get_env(env_name, env_kwargs)
        if not self.discrete_actions:
            self.env = ClipAction(self.env)
        if num_env_steps is None:
            self.num_env_steps = self.default_env_params.max_steps_in_episode
        else:
            self.num_env_steps = num_env_steps

    # --- ENVIRONMENT RESET ---
    def batch_reset(self, rng_reset, env_params):
        """Reset a single environment over a batch of seeds."""
        batch_reset = jax.vmap(self.single_reset, in_axes=(0, None))
        return batch_reset(rng_reset, env_params)

    def single_reset(self, rng_reset, env_params):
        """Reset and environment, returning initial state and observation."""
        obs, state = self.env.reset(rng_reset, env_params)
        return obs, state

    # --- ENVIRONMENT ROLLOUT ---
    def batch_rollout(self, rng_eval, agent_state, env_params, init_obs, init_state):
        """Rollout an agent on a single environment over a batch of seeds and environment states."""
        batch_rollout = jax.vmap(self.single_rollout, in_axes=(0, None, None, 0, 0))
        return batch_rollout(rng_eval, agent_state, env_params, init_obs, init_state)

    def single_rollout(
        self, rng_episode, agent_state, env_params, init_obs, init_state
    ):
        """Rollout an agent on a single environment."""

        def _env_step(runner_state, _):
            train_state, env_state, last_obs, rng = runner_state
            rng, _rng = jax.random.split(rng)
            pi, value = agent_state.apply_fn(agent_state.params, last_obs)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
            rng, _rng = jax.random.split(rng)
            obsv, env_state, reward, done, info = self.env.step(
                _rng, env_state, action, env_params
            )
            transition = Transition(
                done, action, value, reward, log_prob, last_obs, obsv, info
            )
            runner_state = (train_state, env_state, obsv, rng)
            return runner_state, transition

        # Scan over episode step loop
        carry_out, scan_out = jax.lax.scan(
            _env_step,
            (
                agent_state,
                init_state,
                init_obs,
                rng_episode,
            ),
            None,
            self.num_env_steps,
        )
        end_state = carry_out[1]
        end_obs = carry_out[2]
        return end_state, end_obs, scan_out

    @property
    def obs_shape(self):
        """Get the shape of the observation."""
        return self.env.observation_space(self.default_env_params).shape

    @property
    def discrete_actions(self):
        """Check if the action space is discrete."""
        return isinstance(
            self.env.action_space(self.default_env_params), spaces.Discrete
        )

    @property
    def num_actions(self):
        """Get the dimension of the action space."""
        if self.discrete_actions:
            return self.env.action_space(self.default_env_params).n
        return self.env.action_space(self.default_env_params).shape[0]

    @property
    def action_lims(self):
        """Get the action limits for the environment."""
        if self.discrete_actions:
            return None
        return (
            self.env.action_space(self.default_env_params).low,
            self.env.action_space(self.default_env_params).high,
        )

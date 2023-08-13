import jax
import sys

from experiments.parse_args import parse_args
from experiments.logging import init_logger, log_results
from util import *
from agents.agents import get_agent
from environments.rollout import RolloutWrapper


def make_train(args):
    def train(rng):
        # --- Initialize environment ---
        env = RolloutWrapper(args.env_name, args.num_rollout_steps)
        env_params = env.default_env_params
        rng, _rng = jax.random.split(rng)
        _rng = jax.random.split(_rng, args.num_env_workers)
        obsv, env_state = env.batch_reset(_rng, env_params)

        # --- Initialize agent train states and step function ---
        rng, _rng = jax.random.split(rng)
        # train_state contains actor (and critic if used) to be used for rollouts,
        # aux_train_states contains all other trainable parameters
        train_state, aux_train_states, agent_train_step_fn = get_agent(
            args,
            _rng,
            env.obs_shape,
            env.num_actions,
            env.discrete_actions,
            env.action_lims,
        )

        # --- Execute train loop ---
        def _train_step(runner_state, _):
            train_state, aux_train_states, env_state, last_obs, rng = runner_state
            # --- Collect trajectories ---
            rng, _rng = jax.random.split(rng)
            _rng = jax.random.split(_rng, args.num_env_workers)
            new_env_state, new_last_obs, traj_batch = env.batch_rollout(
                _rng, train_state, env_params, last_obs, env_state
            )

            # --- Update agent ---
            rng, _rng = jax.random.split(rng)
            train_state, aux_train_states, loss, metric = agent_train_step_fn(
                train_state, aux_train_states, traj_batch, new_last_obs, _rng
            )

            runner_state = (
                train_state,
                aux_train_states,
                new_env_state,
                new_last_obs,
                rng,
            )
            return runner_state, (loss, metric, traj_batch)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, aux_train_states, env_state, obsv, _rng)
        runner_state, (loss, metric, traj_batch) = jax.lax.scan(
            _train_step, runner_state, None, args.num_train_steps
        )
        ret = {
            "runner_state": runner_state,
            "metrics": metric,
            "loss": loss,
        }
        return ret

    return train


def train_agents(args):
    # --- Initialize experiment ---
    init_logger(args)
    train_fn = make_train(args)

    # --- Run training for num_agents ---
    rng = jax.random.PRNGKey(args.seed)
    rng = jax.random.split(rng, args.num_agents)
    results = jax.vmap(train_fn)(rng)
    log_results(args, results)


def main(cmd_args=sys.argv[1:]):
    args = parse_args(cmd_args)
    experiment_fn = jax_debug_wrapper(args, train_agents)
    return experiment_fn(args)


if __name__ == "__main__":
    main()

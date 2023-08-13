import jax
from flax.training.train_state import TrainState

from models.optimizers import create_optimizer


def _get_agent_networks(args, num_actions, discrete_actions, action_lims):
    # Discrete action space
    if discrete_actions:
        if args.agent == "ppo":
            from models.actor_critic import ActorCritic

            return ActorCritic(num_actions, activation=args.activation), None
        elif args.agent == "dqn":
            from models.q_network import QNetwork

            return QNetwork(num_actions, activation=args.activation), None

        raise ValueError(f"Unknown agent {args.agent} for discrete action space.")

    # Continuous action space
    else:
        if args.agent == "ppo":
            from models.actor_critic import ActorCriticContinuous

            return ActorCriticContinuous(num_actions, activation=args.activation), None
        elif args.agent == "sac":
            from models.soft_actor_critic import (
                TanhGaussianActor,
                VectorCritic,
                EntropyCoef,
            )

            auxilary_networks = (
                VectorCritic(
                    activation=args.activation, n_critics=args.sac_n_critics
                ),  # Q network
                VectorCritic(
                    activation=args.activation, n_critics=args.sac_n_critics
                ),  # Target Q network
                EntropyCoef(),  # Entropy coefficient
            )
            return (
                TanhGaussianActor(
                    num_actions, activation=args.activation, action_lims=action_lims
                ),
                auxilary_networks,
            )

    raise ValueError(f"Unknown agent {args.agent} for continuous action space.")


def _create_agent_train_state(rng, network, args, obs_shape=None, num_actions=None):
    network_init_args = network.init_args(obs_shape, num_actions)
    network_params = network.init(rng, *network_init_args)
    tx = create_optimizer(args)
    return TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )


def _make_train_step(args, network, aux_networks, discrete_actions):
    if discrete_actions:
        if args.agent == "ppo":
            from agents.ppo import make_train_step

            return make_train_step(args, network)
        elif args.agent == "dqn":
            from agents.dqn import make_train_step

            return make_train_step(args, network)
        raise ValueError(f"Unknown agent {args.agent} for discrete action space.")
    if args.agent == "ppo":
        from agents.ppo import make_train_step

        return make_train_step(args, network)
    elif args.agent == "sac":
        from agents.sac import make_train_step

        return make_train_step(args, network, aux_networks)
    raise ValueError(f"Unknown agent {args.agent} for continuous action space.")


def get_agent(args, rng, obs_shape, num_actions, discrete_actions, action_lims):
    """
    Returns the actor-critic network, auxiliary networks/parameters, and the train step function.
    """
    agent_network, aux_networks = _get_agent_networks(
        args, num_actions, discrete_actions, action_lims
    )
    rng, _rng = jax.random.split(rng)
    agent_train_state = _create_agent_train_state(_rng, agent_network, args, obs_shape)
    if aux_networks is not None:
        rng = jax.random.split(rng, len(aux_networks))
        aux_train_states = tuple(
            [
                _create_agent_train_state(
                    key, aux_network, args, obs_shape, num_actions
                )
                for key, aux_network in zip(rng, aux_networks)
            ]
        )
    else:
        aux_train_states = None
    agent_train_step_fn = _make_train_step(
        args, agent_network, aux_networks, discrete_actions
    )
    return agent_train_state, aux_train_states, agent_train_step_fn

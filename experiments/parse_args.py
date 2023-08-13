import sys
import argparse


def parse_args(cmd_args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Debug mode (disable JIT)")
    parser.add_argument(
        "--debug_nans",
        action="store_true",
        help="Exit and stack trace when NaNs are encountered",
    )

    # Experiment
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--num_agents", type=int, default=1, help="Number of agents to train"
    )
    parser.add_argument(
        "--num_train_steps", type=int, default=100, help="Number agent train steps"
    )

    # Environment
    parser.add_argument(
        "--env_name", type=str, default="CartPole-v1", help="Environment name"
    )
    parser.add_argument(
        "--num_env_workers", type=int, default=16, help="Number of environment workers"
    )

    # Agent
    parser.add_argument("--agent", type=str, default="ppo", help="Agent type")
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        help="Activation function for actor critic",
    )
    parser.add_argument(
        "--num_rollout_steps",
        type=int,
        default=128,
        help="Number of rollout steps per agent update",
    )
    parser.add_argument(
        "--num_minibatches", type=int, default=4, help="Number of minibatches"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument(
        "--value_loss_coef", type=float, default=0.5, help="Value loss coefficient"
    )
    parser.add_argument(
        "--entropy_coef", type=float, default=0.01, help="Entropy coefficient"
    )

    # Optimization
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--anneal_lr", action="store_true", help="Anneal learning rate")
    parser.add_argument(
        "--max_grad_norm", type=float, default=0.5, help="Max gradient norm"
    )

    # PPO
    parser.add_argument(
        "--ppo_num_epochs", type=int, default=4, help="Number of epochs"
    )
    parser.add_argument(
        "--ppo_clip_eps", type=float, default=0.2, help="PPO clip epsilon"
    )

    # SAC
    parser.add_argument(
        "--sac_n_critics", type=int, default=2, help="Number of critics"
    )
    parser.add_argument(
        "--sac_polyak_step_size",
        type=float,
        default=0.005,
        help="Target update step size",
    )
    parser.add_argument(
        "--sac_target_entropy", type=float, default=-1.0, help="Target entropy"
    )

    # Logging
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--wandb_project", type=str, help="Wandb project")
    parser.add_argument("--wandb_entity", type=str, help="Wandb entity")
    parser.add_argument("--wandb_group", type=str, default="debug", help="Wandb group")

    args, rest_args = parser.parse_known_args(cmd_args)
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")
    return args

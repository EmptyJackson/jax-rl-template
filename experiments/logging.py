import wandb
import jax.numpy as jnp


def init_logger(args):
    if args.log:
        wandb.init(
            config=args,
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            job_type="train_agents",
        )


def log_results(args, results):
    rets = results["metrics"]["returned_episode_returns"]

    # Hack to avoid logging 0 before first episodes are done
    # Required until https://github.com/RobertTLange/gymnax/issues/62 is resolved
    returned = results["metrics"]["returned_episode"]
    num_agents, num_train_steps, num_env_workers, num_rollout_steps = returned.shape
    first_episode_done = jnp.zeros((num_agents, num_env_workers), dtype=jnp.bool_)
    all_done_step = 0
    while not first_episode_done.all():
        step_episodes_done = jnp.any(returned[:, all_done_step], axis=-1)
        first_episode_done |= step_episodes_done
        all_done_step += 1
    return_list = [
        rets[:, step].mean() for step in range(all_done_step, num_train_steps)
    ]

    print("Step returns:", jnp.around(jnp.array(return_list), decimals=2))
    if args.log:
        for step in range(rets.shape[1]):
            step_ret = None
            if step >= all_done_step:
                step_ret = return_list[step - all_done_step]
            wandb.log(
                {
                    "return": step_ret,
                    "step": step,
                    **{k: v[:, step].mean() for k, v in results["loss"].items()},
                }
            )

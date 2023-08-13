import optax
from functools import partial


def _linear_schedule(count, args):
    if args.agent == "ppo":
        frac = (
            1.0
            - (count // (args.ppo_num_minibatches * args.ppo_update_epochs))
            / args.num_train_steps
        )
    else:
        frac = 1.0 - (count / args.num_train_steps)
    return args.lr * frac


def create_optimizer(args):
    if args.anneal_lr:
        return optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adam(learning_rate=partial(_linear_schedule, args=args), eps=1e-5),
        )
    return optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adam(args.lr, eps=1e-5),
    )

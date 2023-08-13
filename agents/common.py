import jax
import jax.numpy as jnp


def construct_minibatches(rng, args, batch):
    batch_size = args.num_rollout_steps * args.num_env_workers
    assert (
        batch_size % args.num_minibatches == 0
    ), "num_minibatches must be a factor of num_rollout_steps * num_env_workers"
    permutation = jax.random.permutation(rng, batch_size)
    batch = jax.tree_util.tree_map(
        lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
    )
    shuffled_batch = jax.tree_util.tree_map(
        lambda x: jnp.take(x, permutation, axis=0), batch
    )
    return jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, [args.num_minibatches, -1, *x.shape[1:]]),
        shuffled_batch,
    )


def calculate_gae(args, traj_batch, last_val):
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        done, value, reward = (
            transition.done,
            transition.value,
            transition.reward,
        )
        delta = reward + args.gamma * next_value * (1 - done) - value
        gae = delta + args.gamma * args.gae_lambda * (1 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        traj_batch,
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + traj_batch.value

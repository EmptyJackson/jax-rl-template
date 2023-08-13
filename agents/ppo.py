import jax
import jax.numpy as jnp

from agents.common import construct_minibatches, calculate_gae


def make_train_step(args, network):
    def _update_step(train_state, aux_train_states, traj_batch, last_obs, rng):
        def _update_epoch(update_state, _):
            train_state, traj_batch, advantages, targets, rng = update_state

            def _update_minbatch(train_state, batch_info):
                # --- Update agent ---
                traj_batch, advantages, targets = batch_info

                def _loss_fn(params, traj_batch, gae, targets):
                    pi, value = network.apply(params, traj_batch.obs)
                    log_prob = pi.log_prob(traj_batch.action)

                    # Value loss
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-args.ppo_clip_eps, args.ppo_clip_eps)
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = (
                        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    )

                    # Actor loss
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    actor_loss1 = ratio * gae
                    actor_loss2 = (
                        jnp.clip(
                            ratio,
                            1.0 - args.ppo_clip_eps,
                            1.0 + args.ppo_clip_eps,
                        )
                        * gae
                    )
                    actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()
                    entropy = pi.entropy().mean()

                    total_loss = (
                        actor_loss
                        + args.value_loss_coef * value_loss
                        - args.entropy_coef * entropy
                    )
                    return total_loss, (value_loss, actor_loss, entropy)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                (total_loss, (value_loss, actor_loss, entropy)), grads = grad_fn(
                    train_state.params, traj_batch, advantages, targets
                )
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, {
                    "value_loss": value_loss,
                    "actor_loss": actor_loss,
                    "entropy": entropy,
                }

            # --- Iterate over minibatches ---
            batch = (traj_batch, advantages, targets)
            rng, _rng = jax.random.split(rng)
            minibatches = construct_minibatches(_rng, args, batch)
            train_state, loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
            update_state = (train_state, traj_batch, advantages, targets, rng)
            return update_state, loss

        # --- Calculate advantage ---
        _, last_val = network.apply(train_state.params, last_obs)
        advantages, targets = jax.vmap(calculate_gae, in_axes=(None, 0, 0))(
            args, traj_batch, last_val
        )

        # --- Iterate over epochs ---
        update_state = (train_state, traj_batch, advantages, targets, rng)
        update_state, metrics = jax.lax.scan(
            _update_epoch, update_state, None, args.ppo_num_epochs
        )
        train_state = update_state[0]
        info = traj_batch.info
        metrics = jax.tree_map(lambda x: x.mean(), metrics)

        # No auxiliary networks
        return train_state, aux_train_states, metrics, info

    return _update_step

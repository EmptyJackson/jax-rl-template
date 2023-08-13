import jax
import jax.numpy as jnp

from agents.common import construct_minibatches


def make_train_step(args, network):
    def _update_step(train_state, aux_train_states, traj_batch, last_obs, rng):
        # --- Calculate targets ---
        reward, next_obs = (traj_batch.reward, traj_batch.next_obs)
        _, next_vals = network.apply(train_state.params, next_obs)
        targets = reward + args.gamma * jnp.max(next_vals, axis=-1) * (
            1 - traj_batch.done
        )

        # --- Update agent ---
        def _update_minbatch(train_state, batch_info):
            traj_batch, targets = batch_info

            def _loss_fn(params, traj_batch, targets):
                # Standard DQN loss
                _, all_action_values = network.apply(params, traj_batch.obs)
                taken_action_values = jnp.squeeze(
                    jnp.take_along_axis(
                        all_action_values,
                        jnp.expand_dims(traj_batch.action, 1),
                        axis=-1,
                    ),
                    axis=-1,
                )
                return jnp.square(taken_action_values - targets).mean()

            grad_fn = jax.value_and_grad(_loss_fn)
            loss, grads = grad_fn(train_state.params, traj_batch, targets)
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, {"loss": loss}

        batch = (traj_batch, targets)
        rng, _rng = jax.random.split(rng)
        minibatches = construct_minibatches(_rng, args, batch)
        train_state, metrics = jax.lax.scan(_update_minbatch, train_state, minibatches)
        info = traj_batch.info
        # No auxiliary networks
        return train_state, aux_train_states, metrics, info

    return _update_step

import jax
import optax
import jax.numpy as jnp

from agents.common import construct_minibatches


def make_train_step(args, network, aux_networks):
    q_network, _, alpha_network = aux_networks

    def _update_step(train_state, aux_train_states, traj_batch, last_obs, rng):
        def _update_minbatch(runner_state, traj_batch):
            train_state, aux_train_states, rng = runner_state
            q_train_state, q_target_train_state, alpha_train_state = aux_train_states

            # --- Update target networks ---
            new_target_params = jax.tree_map(
                lambda x, y: jnp.where(q_target_train_state.step == 0, x, y),
                q_train_state.params,
                optax.incremental_update(
                    q_train_state.params,
                    q_target_train_state.params,
                    args.sac_polyak_step_size,
                ),
            )
            q_target_train_state = q_target_train_state.replace(
                step=q_target_train_state.step + 1,
                params=new_target_params,
            )

            # --- Compute targets ---
            alpha = alpha_network.apply(alpha_train_state.params)

            def _compute_target(rng, transition):
                next_pi, _ = network.apply(train_state.params, transition.next_obs)
                # Note: Important to use sample_and_log_prob here, not just sample, for numerical stability
                # See https://github.com/deepmind/distrax/issues/7
                next_action, log_next_pi = next_pi.sample_and_log_prob(seed=rng)
                # Minimum of the target Q-values
                next_q_value = jnp.min(
                    q_network.apply(
                        q_target_train_state.params, transition.next_obs, next_action
                    )
                )
                target = transition.reward + args.gamma * (1 - transition.done) * (
                    next_q_value - alpha * log_next_pi
                )
                return target

            rng, _rng = jax.random.split(rng)
            _rng = jax.random.split(_rng, traj_batch.obs.shape[0])
            targets = jax.vmap(_compute_target)(_rng, traj_batch)

            # --- Update critics ---
            def _q_loss_fn(params):
                # Compute loss for all critics
                q_pred = q_network.apply(params, traj_batch.obs, traj_batch.action)
                return jnp.square(q_pred - targets).mean()

            critic_loss, critic_grad = jax.value_and_grad(_q_loss_fn)(
                q_train_state.params
            )
            q_train_state = q_train_state.apply_gradients(grads=critic_grad)

            # --- Update actor ---
            def _actor_loss_function(params, rng):
                def _transition_loss(rng, transition):
                    pi, _ = network.apply(params, transition.obs)
                    sampled_action, log_pi = pi.sample_and_log_prob(seed=rng)
                    q_values = q_network.apply(
                        q_train_state.params, transition.obs, sampled_action
                    )
                    return -jnp.min(q_values) + alpha * log_pi, -log_pi

                rng, _rng = jax.random.split(rng)
                _rng = jax.random.split(_rng, traj_batch.obs.shape[0])
                loss, entropy = jax.vmap(_transition_loss)(_rng, traj_batch)
                return loss.mean(), entropy.mean()

            rng, _rng = jax.random.split(rng)
            (actor_loss, entropy), actor_grad = jax.value_and_grad(
                _actor_loss_function, has_aux=True
            )(train_state.params, _rng)
            train_state = train_state.apply_gradients(grads=actor_grad)

            # --- Update alpha ---
            def _alpha_loss_fn(params):
                alpha_value = alpha_network.apply(params)
                return alpha_value * (entropy - args.sac_target_entropy).mean()

            alpha_grad = jax.grad(_alpha_loss_fn)(alpha_train_state.params)
            alpha_train_state = alpha_train_state.apply_gradients(grads=alpha_grad)
            aux_train_states = q_train_state, q_target_train_state, alpha_train_state
            metrics = {
                "critic_loss": critic_loss,
                "actor_loss": actor_loss,
                "entropy": entropy,
                "alpha": alpha_network.apply(alpha_train_state.params),
            }
            return (train_state, aux_train_states, rng), metrics

        # --- Iterate over minibatches ---
        rng, _rng = jax.random.split(rng)
        minibatches = construct_minibatches(_rng, args, traj_batch)
        (train_state, aux_train_states, _), metrics = jax.lax.scan(
            _update_minbatch,
            (train_state, aux_train_states, rng),
            minibatches,
        )
        info = traj_batch.info
        metrics = jax.tree_map(lambda x: x.mean(), metrics)
        return (
            train_state,
            aux_train_states,
            metrics,
            info,
        )

    return _update_step

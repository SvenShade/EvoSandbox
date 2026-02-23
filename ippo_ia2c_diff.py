--- ff_ippo.py
+++ ff_ippo.py
@@ -257,6 +257,12 @@
                 params, opt_states, key = train_state
                 traj_batch, advantages, targets = batch_info
 
+                # Switch between PPO-style clipped objective and IA2C/A2C objective.
+                # Default is PPO clipping for backwards compatibility.
+                use_ppo_clipping = getattr(config.system, "use_ppo_clipping", True)
+                clip_eps = getattr(config.system, "clip_eps", 0.2)
+
+
                 def _actor_loss_fn(
                     actor_params: FrozenDict,
                     traj_batch: PPOTransition,
@@ -270,19 +276,23 @@
 
                     # Calculate actor loss
                     ratio = jnp.exp(log_prob - traj_batch.log_prob)
-                    # Nomalise advantage at minibatch level
+
+                    # Normalise advantage at minibatch level (works for both PPO and A2C/IA2C).
                     gae = (gae - gae.mean()) / (gae.std() + 1e-8)
-                    actor_loss1 = ratio * gae
-                    actor_loss2 = (
-                        jnp.clip(
+
+                    if use_ppo_clipping:
+                        # PPO clipped surrogate objective.
+                        actor_loss1 = ratio * gae
+                        actor_loss2 = jnp.clip(
                             ratio,
-                            1.0 - config.system.clip_eps,
-                            1.0 + config.system.clip_eps,
-                        )
-                        * gae
-                    )
-                    actor_loss = -jnp.minimum(actor_loss1, actor_loss2)
-                    actor_loss = actor_loss.mean()
+                            1.0 - clip_eps,
+                            1.0 + clip_eps,
+                        ) * gae
+                        actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()
+                    else:
+                        # IA2C/A2C: vanilla policy-gradient loss (on-policy).
+                        # Note: ratio is kept for logging/diagnostics; it is not used in the objective.
+                        actor_loss = -(log_prob * gae).mean()
                     # The seed will be used in the TanhTransformedDistribution:
                     entropy = actor_policy.entropy(seed=key).mean()
 
@@ -298,13 +308,17 @@
                     # Rerun network
                     value = critic_apply_fn(critic_params, traj_batch.obs)
 
-                    # Clipped MSE loss
-                    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
-                        -config.system.clip_eps, config.system.clip_eps
-                    )
-                    value_losses = jnp.square(value - targets)
-                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
-                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
+                    if use_ppo_clipping:
+                        # PPO-style clipped value loss.
+                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
+                            -clip_eps, clip_eps
+                        )
+                        value_losses = jnp.square(value - targets)
+                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
+                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
+                    else:
+                        # IA2C/A2C: standard MSE value regression.
+                        value_loss = 0.5 * jnp.square(value - targets).mean()
 
                     total_value_loss = config.system.vf_coef * value_loss
                     return total_value_loss, value_loss

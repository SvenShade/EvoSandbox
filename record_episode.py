from typing import Callable, List, Optional          # ← add “Callable, List, Optional”
try:
    # If you already have a bespoke render() util, this will just work.
    from render import render                        # noqa: E402
except ModuleNotFoundError:
    # Fallback so the script still runs even if no renderer is present yet.
    def render(env_state):                           # type: ignore
        """Stub ‑ replace with your own implementation."""
        pass


@@ def learner_setup(...):
-    # Initialise observation with obs of all agents.
-    obs = env.observation_spec.generate_value()
-    init_x = tree.map(lambda x: x[jnp.newaxis, ...], obs)      # ❌ adds wrong axis
+    # Initialise observation with obs of all agents.
+    obs          = env.observation_spec.generate_value()       # (num_agents, obs_dim)
+    num_envs     = config.arch.num_envs
+    # Broadcast so the shape matches run‑time: (num_envs, num_agents, obs_dim)
+    init_x       = tree.map(lambda x: jnp.broadcast_to(x, (num_envs,) + x.shape), obs)


def _render_single_episode(
    env: MarlEnv,
    actor_apply_fn: ActorApply,
    actor_params: FrozenDict,
    seed: int = 0,
    render_fn: Optional[Callable[[Any], None]] = None,
) -> List[Any]:
    """
    Roll one episode with *deterministic* actions from the trained actor and
    call `render_fn(env_state)` (default: `render`) each step.

    Returns a list of whatever `render_fn` returns (e.g. RGB frames).
    """
    if render_fn is None:
        render_fn = render

    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    env_state, timestep = env.reset(reset_key)

    frames = []
    while not timestep.last().all():
        # Add batch dim expected by the actor network.
        batched_obs = tree.map(lambda x: x[jnp.newaxis, ...], timestep.observation)
        actor_policy = actor_apply_fn(actor_params, batched_obs)
        # Use mode() for deterministic evaluation
        action = actor_policy.mode()
        # Remove leading batch dim again
        env_state, timestep = env.step(env_state, action.squeeze(0))
        frames.append(render_fn(env_state))
    return frames


# Use the highest‑return params if we tracked them, otherwise the final ones.
render_params = (
    best_params if (config.arch.absolute_metric and best_params is not None)
    else unreplicate_batch_dim(learner_state.params.actor_params)
)

# Fresh (non‑vectorised) env for rendering so we get one clean episode.
render_env, _ = environments.make(config)
_ = _render_single_episode(           # noqa: F841  (frames are returned if you need them)
    render_env,
    actor_network.apply,
    render_params,
    seed=config.system.seed + 1,      # use a different seed from training
)

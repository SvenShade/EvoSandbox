# ---------------------------------------------------------------------------
# Single-episode roll-out with per-timestep rendering
# ---------------------------------------------------------------------------
from typing import Optional
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from mava.types import ActorApply, MarlEnv
# NOTE: `render` must be supplied elsewhere (e.g. your own plotting / video writer).

def play_and_render_episode(
    env: MarlEnv,
    params: FrozenDict,
    actor_apply_fn: ActorApply,
    config,
    seed: int = 0,
) -> None:
    """
    Roll a *single* un-batched episode to termination and call `render(state)`
    at every step.

    Args
    ----
    env:           A **single-environment** instance (no vectorisation).
    params:        The *actor* parameter PyTree to execute.
    actor_apply_fn:The usual `Actor.apply`.
    config:        Full Hydra config so we can respect `evaluation_greedy`.
    seed:          RNG seed for the episode (default 0).
    """
    key = jax.random.PRNGKey(seed)

    # ------------------------------------------------------------------ #
    # Reset                                                               #
    # ------------------------------------------------------------------ #
    reset_out = env.reset(seed=seed)
    # Some envs return (state, timestep), others only timestep.
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        env_state, ts = reset_out                       # Jumanji-style :contentReference[oaicite:0]{index=0}
    else:
        env_state, ts = None, reset_out                 # Gym-style (internal state)

    done = bool(ts.last())
    actor_state = {}  # not needed for feed-forward policy, kept for API symmetry

    # ------------------------------------------------------------------ #
    # Episode loop                                                        #
    # ------------------------------------------------------------------ #
    while not done:
        key, act_key = jax.random.split(key)

        # Greedy vs stochastic exactly as in evaluator :contentReference[oaicite:1]{index=1}
        pi = actor_apply_fn(params, ts.observation)
        action = pi.mode() if config.arch.evaluation_greedy else pi.sample(seed=act_key)

        # Step the environment (handle stateful & stateless variants)
        if env_state is None:
            ts = env.step(jax.device_get(action))
        else:
            env_state, ts = env.step(env_state, action)

        # ----------------------------------------------------------------
        # User-supplied frame renderer
        # ----------------------------------------------------------------
        frame_state = env_state if env_state is not None else env  # fallback
        render(jax.device_get(frame_state))  # <-- implement elsewhere

        done = bool(ts.last())


# ─────────────────────────────────────────────────────────────────────
# Post-training visual roll-out
# add before logger.stop() in run_experiment
# ─────────────────────────────────────────────────────────────────────
play_params = best_params if best_params is not None else trained_params

# Make a fresh, *unbatched* env (one copy only) for rendering
render_env, _ = environments.make(config)     # same helper you already use
render_env.unwrapped.num_envs = 1             # defensively ensure singular

play_and_render_episode(
    render_env,
    play_params,
    actor_network.apply,
    config,
    seed=config.system.seed + 12345,          # arbitrary, reproducible
)

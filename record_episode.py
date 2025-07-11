# ---------------------------------------------------------------------------
# Single-episode roll-out with per-timestep rendering
# ---------------------------------------------------------------------------
from typing import Optional
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from mava.types import ActorApply, MarlEnv

def play_and_render_episode(
    env: MarlEnv,
    params: FrozenDict,          # trained *actor* params
    actor_apply_fn,
    config,
    seed: int = 0,
):
    # Make the same act-fn the evaluator uses (greedy vs. sample handled inside)
    act_fn = make_ff_eval_act_fn(actor_apply_fn, config)

    key = jax.random.PRNGKey(seed)
    ts = env.reset(seed=seed)          # single unbatched env
    done = bool(ts.last())

    while not done:
        
        obs = ts.observation
        if obs.ndim == 2:
            obs = obs[None, ...]

        key, act_key = jax.random.split(key)
        action, _ = act_fn(params, ts, act_key, {})     # same logic, no actor_state

        if action.ndim == 3 and action.shape[0] == 1:
            action = jnp.squeeze(action, axis=0)
        
        ts = env.step(jax.device_get(action))           # cpu action → env
        render(ts)                                      # ← implement this
        done = bool(ts.last())


# ─────────────────────────────────────────────────────────────────────
# Post-training visual roll-out
# add before logger.stop() in run_experiment
# ─────────────────────────────────────────────────────────────────────
single_env, _ = environments.make(config)   # un-vectorised
play_and_render_episode(
    single_env,
    best_params if config.arch.absolute_metric else trained_params,
    actor_network.apply,
    config,
    seed=config.system.seed + 42,
)

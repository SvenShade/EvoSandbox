# ────────────────────────────────────────────────────────────────────────────────
# GIF-recording utility
# ────────────────────────────────────────────────────────────────────────────────

# existing imports …
import copy
import time
from typing import Any, Tuple

+ from pathlib import Path
+ import imageio.v2 as imageio          # <─ modern imageio API
+ import numpy as np

import chex
import flax


def record_episode_gif(
    env: MarlEnv,
    actor_apply_fn: ActorApply,
    actor_params: FrozenDict,
    *,
    max_steps: int,
    fps: int,
    seed: int,
    out_path: Path,
    greedy: bool = True,          # NEW ─ follow arch.evaluation_greedy
) -> None:
    …
    key = jax.random.PRNGKey(seed)
    env_state, timestep = env.reset(key)

    frames: list[np.ndarray] = []
    frames.append(np.asarray(env.render(env_state)).astype(np.uint8))

    step, done = 0, jnp.any(timestep.last())
    while (not done) and step < max_steps:
         policy_dist = actor_apply_fn(actor_params, timestep.observation)
         if greedy:
             action = policy_dist.mode()            # deterministic
         else:                                      # stochastic like Quick-start
             key, a_key = jax.random.split(key)
             action = policy_dist.sample(seed=a_key)

        env_state, timestep = env.step(env_state, action)
        frames.append(np.asarray(env.render(env_state)).astype(np.uint8))

        done, step = jnp.any(timestep.last()), step + 1



# Insert before return_eval():

# ────────────────────────────────────────────────────────────────────
# [NEW]  Render a single evaluation episode to GIF (optional)
# ────────────────────────────────────────────────────────────────────
if config.env.get("render_gif", False):
    gif_path = Path(logger.log_dir) / config.env.gif_path
    record_episode_gif(
        eval_env,
        actor_network.apply,
        best_params if config.arch.absolute_metric else trained_params,
        max_steps=config.env.gif_max_steps,
        fps=config.env.gif_fps,
        seed=config.system.seed + 42,
        out_path=gif_path,
        greedy=config.arch.evaluation_greedy
    )


# add to ff_ippo.yaml:
env:
  render_gif: true          # turn feature on/off
  gif_path: "episode.gif"   # relative to logger.log_dir
  gif_fps: 12
  gif_max_steps: 2048       # safeguard for very long episodes

arch:
  evaluation_greedy: false   # sample actions during GIF recording


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
+ import matplotlib
+ matplotlib.use("Agg")               # head-less backend for servers / TPUs
+ import matplotlib.pyplot as plt
+ from matplotlib.patches import Circle

import chex
import flax


# ───────────────────────────────────────────────────────────────────────────────
# MPE-style single-frame renderer (borrowed from mpe_visualizer.py)
# ───────────────────────────────────────────────────────────────────────────────
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def _attach_mpe_render(env):
    """Monkey-patch `env.render(state)` onto an MPE environment.

    The first call lazily builds the matplotlib artists; subsequent calls just
    update them and grab the RGB buffer → np.ndarray[H,W,3] (uint8)."""
    # Cache objects on the env instance itself so they persist across calls
    env._render_setup_done = False

    def _render(state):
        # 1️⃣ Lazy initialisation ------------------------------------------------
        if not env._render_setup_done:
            # Figure / axes
            env._fig, env._ax = plt.subplots(figsize=(5, 5), dpi=100)
            env._ax.set_xlim([-2, 2])
            env._ax.set_ylim([-2, 2])
            env._ax.set_aspect("equal")
            env._ax.axis("off")

            # Entity circles
            env._entity_patches = []
            for i in range(env.num_entities):
                patch = Circle(
                    (0, 0),
                    radius=env.rad[i],
                    color=np.asarray(env.colour[i]) / 255.0,
                )
                env._ax.add_patch(patch)
                env._entity_patches.append(patch)

            # Step counter text
            env._step_txt = env._ax.text(-1.9, 1.9, "", va="top", fontsize=9)

            # Optional comm text
            env._comm_idx = np.where(env.silent == 0)[0] if not np.all(env.silent) else []
            env._comm_txt = []
            for j, idx in enumerate(env._comm_idx):
                txt = env._ax.text(-1.9, -1.9 + j * 0.17, "", fontsize=8)
                env._comm_txt.append(txt)

            env._render_setup_done = True

        # 2️⃣ Update artists -----------------------------------------------------
        for i, patch in enumerate(env._entity_patches):
            patch.center = state.p_pos[i]

        env._step_txt.set_text(f"Step: {state.step}")

        for j, idx in enumerate(env._comm_idx):
            letter = ALPHABET[np.argmax(state.c[idx])]
            env._comm_txt[j].set_text(f"{env.agents[idx]} sends {letter}")

        # 3️⃣ Rasterise and return np array --------------------------------------
        env._fig.canvas.draw()
        w, h = env._fig.canvas.get_width_height()
        rgb = np.frombuffer(env._fig.canvas.tostring_rgb(), dtype=np.uint8)
        return rgb.reshape(h, w, 3)

    # Bind method
    import types
    env.render = types.MethodType(_render, env)


# Attach head-less renderers so record_episode_gif() works, immediately after env, eval_env = environments.make(config)
_attach_mpe_render(env)
_attach_mpe_render(eval_env)


# ────────────────────────────────────────────────────────────────────────────────
# GIF-recording utility
# ────────────────────────────────────────────────────────────────────────────────
def record_episode_gif(
    env: MarlEnv,
    actor_apply_fn: ActorApply,
    actor_params: FrozenDict,
    *,
    max_steps: int,
    fps: int,
    seed: int,
    out_path: Path,
    greedy: bool = True,
) -> None:
    """
    Runs a single episode with `actor_params` in `env` and saves the RGB frames
    to `out_path` as an animated GIF.  Works on CPU; no JIT / grad required.
    """
    # Reset environment
    key = jax.random.PRNGKey(seed)
    env_state, timestep = env.reset(key)

    frames: list[np.ndarray] = []

    # Render the first frame
    frames.append(np.asarray(env.render(env_state)).astype(np.uint8))

    step = 0
    done = jnp.any(timestep.last())

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

    # Write GIF
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"[GIF] Saved episode ({len(frames)} frames) → {out_path.resolve()}")


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


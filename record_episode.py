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


print(jax.tree_util.tree_flatten(policy_params)[1][:20])  # first 20 paths


# Patch env.render(state) onto an environment.
def _attach_mpe_render(env):
    # Cache objects on the env instance itself so they persist across calls
    env._render_setup_done = False
    def _render(self, state):
        # Lazy initialisation.
        if not getattr(self, "_render_setup_done", False):
            # Figure / axes
            self._fig, self._ax = plt.subplots(figsize=(5, 5), dpi=100)
            self._ax.set_xlim([-2, 2])
            self._ax.set_ylim([-2, 2])
            self._ax.set_aspect("equal")
            self._ax.axis("off")
    
            # Entities.
            self._entity_patches = []
            for i in range(self.num_entities):
                patch = Circle(
                    (0, 0),
                    radius=self.rad[i],
                    color=np.asarray(self.colour[i]) / 255.0,
                )
                self._ax.add_patch(patch)
                self._entity_patches.append(patch)
    
            # Step counter.
            self._step_txt = self._ax.text(-1.9, 1.9, "", va="top", fontsize=9)    
            self._render_setup_done = True
    
        # Update artists.
        for i, patch in enumerate(self._entity_patches):
            patch.center = state.p_pos[i]
        self._step_txt.set_text(f"Step: {state.step}")
    
        # Render.
        self._fig.canvas.draw()
        w, h = self._fig.canvas.get_width_height()        
        rgb = np.asarray(self._fig.canvas.buffer_rgba()) # (h*w*4)
        rgb = rgb.reshape(h, w, 4)[..., :3]
        return rgb.copy() 

    # Bind method
    import types
    env.render = types.MethodType(_render, env)


# Attach head-less renderers so record_episode_gif() works, immediately after env, eval_env = environments.make(config)
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
        # Make obs shape (1, num_agents, obs_dim) so it matches training time
        batched_obs = jax.tree_map(lambda x: x[jnp.newaxis, ...], timestep.observation)
        policy_dist = actor_apply_fn(actor_params, batched_obs)
        # policy_dist has a leading env-axis of size 1; strip it off again
        policy_dist = jax.tree_map(lambda d: jax.tree_util.tree_leaves(d)[0]
                                   if hasattr(d, "event_shape") else d[0],
                                   policy_dist)
        
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


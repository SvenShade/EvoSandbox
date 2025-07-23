from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Any
import chex
import jax
import jax.numpy as jnp

# CONFIG

@dataclass(frozen=True)
class EnvConfig:
    n_agents: int
    team_ids: jnp.ndarray               # shape [N], int32 team id per agent
    mass: jnp.ndarray                   # [N]
    moveable: jnp.ndarray               # [N] 0/1
    radius: jnp.ndarray                 # [N]
    max_speed: float
    damping: float
    dt: float

    # Vision / interaction
    vision_radius: float
    tag_radius: float
    fire_radius: float
    min_shooters_to_kill: int = 2

    # Rewards
    r_tag_enemy: float = 1.0
    r_friend_tag_penalty: float = -0.2
    r_collision_penalty: float = -0.05
    r_spread: float = 0.1

    # Battery
    battery_decay_per_step: float = 0.0
    fire_battery_cost: float = 0.0
    battery_kill_on_zero: bool = True

    # Episode
    max_steps: int = 500

    # Debug
    debug_asserts: bool = False


@dataclass
class EnvState:
    step: jnp.ndarray                   # ()
    p_pos: jnp.ndarray                  # [N, 2]
    p_vel: jnp.ndarray                  # [N, 2]
    battery: jnp.ndarray                # [N]
    active: jnp.ndarray                 # [N] bool
    rng_key: jnp.ndarray                # PRNGKey


# UTILITIES

def _upper_tri_sum(mat: jnp.ndarray) -> jnp.ndarray:
    """Sum over upper triangle (i<j) for symmetric pair costs to avoid double counting."""
    n = mat.shape[0]
    mask = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)
    return jnp.sum(jnp.where(mask, mat, 0.0), axis=1)

def _vision_mask(dist_sq: jnp.ndarray, vision_sq: float) -> jnp.ndarray:
    return dist_sq <= vision_sq

def _broadcast_mask(mask_vec: jnp.ndarray) -> jnp.ndarray:
    """Turn [N] into [N,1] for broadcasting."""
    return mask_vec[:, None]

def _apply_max_speed(vel: jnp.ndarray, max_speed: float) -> jnp.ndarray:
    speed = jnp.linalg.norm(vel, axis=-1, keepdims=True)
    factor = jnp.minimum(1.0, max_speed / (speed + 1e-8))
    return vel * factor

# ENVIRONMENT

class SwarmEnv:
    """
    MARL swarm vs swarm environment (JAX-friendly).
    Public API:
        - reset(key) -> (obs_dict, state)
        - step_env(state, action_dict) -> (obs_dict, reward_dict, done_dict, info_dict, next_state)
        - get_obs(state) -> obs_dict
    """

    def __init__(self, config: EnvConfig):
        self.cfg = config
        self.N = config.n_agents
        chex.assert_shape(config.team_ids, (self.N,))
        self.teams = config.team_ids
        self.num_teams = int(jnp.max(self.teams)) + 1

        # Precompute masks
        self.same_team = (self.teams[:, None] == self.teams[None, :])
        self.diff_team = ~self.same_team

        # Observation dimension (document clearly)
        # pos of others (2*N), vel of others (2*N), relative pos/vel optional,
        # battery (1), team onehot (T), active flag (1)
        self.team_onehot = jax.nn.one_hot(self.teams, self.num_teams)

        # We'll compute obs shape dynamically to keep code robust.
        dummy_state = EnvState(
            step=jnp.array(0, dtype=jnp.int32),
            p_pos=jnp.zeros((self.N, 2), dtype=jnp.float32),
            p_vel=jnp.zeros((self.N, 2), dtype=jnp.float32),
            battery=jnp.ones((self.N,), dtype=jnp.float32),
            active=jnp.ones((self.N,), dtype=bool),
            rng_key=jax.random.PRNGKey(0),
        )
        self._obs_example = self._compute_obs_tensor(dummy_state)
        self.obs_dim = self._obs_example.shape[-1]

        # Wrap jit functions
        self._reset_jit = jax.jit(self._reset_impl, static_argnums=0)
        self._step_jit = jax.jit(self._step_impl, static_argnums=0)
        self._obs_jit  = jax.jit(self._compute_obs_tensor, static_argnums=0)

    # ------------- Public API -------------
    def reset(self, rng_key: jnp.ndarray) -> Tuple[Dict[str, jnp.ndarray], EnvState]:
        obs, state = self._reset_jit(rng_key)
        return self._obs_to_dict(obs), state

    def step_env(self, state: EnvState, action_dict: Dict[str, jnp.ndarray]):
        # Convert action dict -> array [N, A]
        actions = self._dict_to_array(action_dict)
        obs, rew, done, info, next_state = self._step_jit(state, actions)
        return self._obs_to_dict(obs), self._rew_to_dict(rew), self._done_to_dict(done), info, next_state

    def get_obs(self, state: EnvState) -> Dict[str, jnp.ndarray]:
        obs = self._obs_jit(state)
        return self._obs_to_dict(obs)

    # ------------- Internals -------------

    def _reset_impl(self, rng_key: jnp.ndarray):
        key_pos, key_vel, key_bat = jax.random.split(rng_key, 3)
        # Simple random spawn; customize as needed
        p_pos = jax.random.uniform(key_pos, (self.N, 2), minval=-1.0, maxval=1.0)
        p_vel = jax.random.normal(key_vel, (self.N, 2)) * 0.0
        battery = jnp.ones((self.N,), dtype=jnp.float32)
        active = jnp.ones((self.N,), dtype=bool)

        state = EnvState(
            step=jnp.array(0, dtype=jnp.int32),
            p_pos=p_pos,
            p_vel=p_vel,
            battery=battery,
            active=active,
            rng_key=rng_key,
        )
        obs = self._compute_obs_tensor(state)
        return obs, state

    def _step_impl(self, state: EnvState, actions: jnp.ndarray):
        """
        actions: [N, 3]
            0: force_x
            1: force_y
            2: fire (binary)
        """
        cfg = self.cfg
        N = self.N

        # ---------- Physics ----------
        p_force = actions[:, :2]  # clamp later if needed
        fire_cmd = (actions[:, 2] > 0.5) & state.active

        # Update velocity (semi-implicit)
        vel = state.p_vel
        mass = cfg.mass[:, None]
        move_mask = _broadcast_mask(cfg.moveable.astype(jnp.float32))
        vel_new = vel * (1.0 - cfg.damping) + (p_force / jnp.maximum(mass, 1e-6)) * cfg.dt * move_mask
        vel_new = _apply_max_speed(vel_new, cfg.max_speed)

        pos_new = state.p_pos + vel_new * cfg.dt * move_mask

        # ---------- Battery ----------
        battery_use = cfg.battery_decay_per_step + cfg.fire_battery_cost * fire_cmd.astype(jnp.float32)
        battery_new = jnp.maximum(state.battery - battery_use, 0.0)

        battery_empty = battery_new <= 0.0
        died_battery = battery_empty & cfg.battery_kill_on_zero

        # ---------- Distances ----------
        diff = pos_new[:, None, :] - pos_new[None, :, :]               # [N,N,2]
        dist_sq = jnp.sum(diff**2, axis=-1)                             # [N,N]
        jnp.fill_diagonal(dist_sq, 1e9)  # avoid self
        vision_sq = cfg.vision_radius ** 2
        in_vision = _vision_mask(dist_sq, vision_sq)

        # ---------- Tag / Collision ----------
        tag_sq = cfg.tag_radius ** 2
        tag_hits = in_vision & (dist_sq <= tag_sq)
        # shooters/victims separation by team
        tag_enemy = tag_hits & self.diff_team

        # Friendly tag penalty (same team only, upper-tri to split cost)
        friend_tag_pairs = tag_hits & self.same_team
        friend_tag_pen = _upper_tri_sum(friend_tag_pairs.astype(jnp.float32)) * cfg.r_friend_tag_penalty

        # Collision penalty (use radii)
        radii_sum = cfg.radius[:, None] + cfg.radius[None, :]
        collide = in_vision & (dist_sq <= (radii_sum ** 2)) & (dist_sq > 0)
        coll_pen = _upper_tri_sum(collide.astype(jnp.float32)) * cfg.r_collision_penalty

        # ---------- Fire / AoE ----------
        fire_sq = cfg.fire_radius ** 2
        fire_mask = fire_cmd[:, None] & in_vision & (dist_sq <= fire_sq)
        # victim wise: count shooters
        shooters_per_victim = jnp.sum(fire_mask.astype(jnp.int32), axis=0)
        killed_by_fire = shooters_per_victim >= cfg.min_shooters_to_kill
        # Only active enemies can be killed
        killed_by_fire = killed_by_fire & state.active & self.diff_team.any(axis=0)  # ensure cross-team
        # Per shooter reward = number of victims they contributed to that actually died (>= threshold)
        victims_mask = jnp.where(killed_by_fire[None, :], 1.0, 0.0)
        coordinated_fire = fire_mask.astype(jnp.float32) * victims_mask
        kill_counts = jnp.sum(coordinated_fire, axis=1)  # per shooter

        # ---------- Update active status ----------
        newly_dead = died_battery | killed_by_fire
        active_new = state.active & (~newly_dead)

        # ---------- Rewards ----------
        # Tag enemy reward: count enemy hits (upper-tri avoided? We give credit per tagger)
        tag_hit_enemy_counts = jnp.sum(tag_enemy.astype(jnp.float32), axis=1)

        spread_reward = self._compute_spread_reward(pos_new, active_new)

        reward = (
            tag_hit_enemy_counts * cfg.r_tag_enemy
            + kill_counts * cfg.r_tag_enemy
            + friend_tag_pen
            + coll_pen
            + spread_reward
        )

        # Mask rewards of inactive agents
        reward = reward * active_new.astype(jnp.float32)

        # ---------- Done logic ----------
        step_next = state.step + 1
        team_alive_counts = jnp.array([
            jnp.sum((self.teams == t) & active_new) for t in range(self.num_teams)
        ])
        team_wiped = team_alive_counts == 0
        done_all = jnp.any(team_wiped) | (step_next >= cfg.max_steps)

        done_vec = jnp.where(active_new, done_all, True)

        info: Dict[str, Any] = {
            "team_alive_counts": team_alive_counts,
            "fire_kills": killed_by_fire,
        }

        next_state = EnvState(
            step=step_next,
            p_pos=pos_new,
            p_vel=vel_new,
            battery=battery_new,
            active=active_new,
            rng_key=state.rng_key,  # not used yet
        )

        obs_next = self._compute_obs_tensor(next_state)

        if self.cfg.debug_asserts:
            chex.assert_shape(obs_next, (self.N, self.obs_dim))
            chex.assert_equal_shape([reward, done_vec, active_new])

        return obs_next, reward, done_vec, info, next_state

    # ----------------- Observations -----------------

    def _compute_obs_tensor(self, state: EnvState) -> jnp.ndarray:
        """
        Per-agent observation vector.
        You can customize; here we include:
          - self pos/vel (4)
          - others' relative pos/vel (2*(N-1)*2)
          - battery (1), active (1)
          - team onehot (T)
        """
        N = self.N
        pos = state.p_pos
        vel = state.p_vel
        rel_pos = pos[None, :, :] - pos[:, None, :]          # [N,N,2]
        rel_vel = vel[None, :, :] - vel[:, None, :]          # [N,N,2]

        mask_others = ~jnp.eye(N, dtype=bool)
        rel_pos = rel_pos[mask_others].reshape(N, N-1, 2)
        rel_vel = rel_vel[mask_others].reshape(N, N-1, 2)

        obs_list = [
            pos,                         # [N,2]
            vel,                         # [N,2]
            rel_pos.reshape(N, -1),      # [N, 2*(N-1)]
            rel_vel.reshape(N, -1),      # [N, 2*(N-1)]
            state.battery[:, None],      # [N,1]
            state.active[:, None].astype(jnp.float32),  # [N,1]
            self.team_onehot.astype(jnp.float32)        # [N,T]
        ]
        return jnp.concatenate(obs_list, axis=-1)

    # ----------------- Rewards helpers -----------------

    def _compute_spread_reward(self, pos: jnp.ndarray, active: jnp.ndarray) -> jnp.ndarray:
        """
        Encourage team spread: reward agents inversely to local density of *their own* team.
        (Simplest proxy: negative of number of same-team neighbors within vision radius)
        """
        cfg = self.cfg
        diff = pos[:, None, :] - pos[None, :, :]
        dist_sq = jnp.sum(diff**2, axis=-1)
        jnp.fill_diagonal(dist_sq, 1e9)
        within = dist_sq <= cfg.vision_radius ** 2
        same = within & self.same_team
        density = jnp.sum(same.astype(jnp.float32), axis=1)
        spread = -density * cfg.r_spread  # lower density => less negative (higher reward)
        return spread * active.astype(jnp.float32)

    # ----------------- Dict helpers -----------------

    def _obs_to_dict(self, obs: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        return {f"agent_{i}": obs[i] for i in range(self.N)}

    def _rew_to_dict(self, rew: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        return {f"agent_{i}": rew[i] for i in range(self.N)}

    def _done_to_dict(self, done: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        out = {f"agent_{i}": done[i] for i in range(self.N)}
        out["__all__"] = jnp.all(done)
        return out

    def _dict_to_array(self, d: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        # assume consistent order
        return jnp.stack([d[f"agent_{i}"] for i in range(self.N)], axis=0)


def make_env(
    n_agents: int,
    team_ids: jnp.ndarray,
    **kwargs,
) -> SwarmEnv:
    cfg = EnvConfig(
        n_agents=n_agents,
        team_ids=team_ids.astype(jnp.int32),
        mass=jnp.ones((n_agents,), dtype=jnp.float32),
        moveable=jnp.ones((n_agents,), dtype=jnp.int32),
        radius=jnp.ones((n_agents,), dtype=jnp.float32) * 0.05,
        max_speed=1.0,
        damping=0.25,
        dt=0.1,
        vision_radius=1.0,
        tag_radius=0.1,
        fire_radius=0.3,
        **kwargs,
    )
    return SwarmEnv(cfg)

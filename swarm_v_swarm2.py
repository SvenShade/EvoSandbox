"""
Swarm-v-Swarm MPE scenario for JaxMARL (single class, no env collision forces).

Action per agent: [ax, ay, fire_bit] in [-1, 1]
  ax, ay -> throttle (scaled by per-agent accel)
  fire_bit > 0.5 -> fire attempt (costs battery; AoE requires >=2 shooters)

Episode ends at max_steps or when one team is wiped.
"""

from functools import partial
from typing import Dict, Tuple, Optional

import chex
import jax
import jax.numpy as jnp
from flax import struct

from jaxmarl.environments.mpe.default_params import DT, MAX_STEPS, DAMPING
from jaxmarl.environments.spaces import Box
from jaxmarl.environments.multi_agent_env import MultiAgentEnv


# --------------------------------------------------------------------------- #
#                                   State                                     #
# --------------------------------------------------------------------------- #

@struct.dataclass
class State:
    p_pos: chex.Array    # [num_entities, 2]
    p_vel: chex.Array    # [num_entities, 2]
    done: chex.Array     # [N]
    step: int
    active: chex.Array   # [N] bool
    battery: chex.Array  # [N] float32


# --------------------------------------------------------------------------- #
#                             Swarm-v-Swarm Env                               #
# --------------------------------------------------------------------------- #

class SwarmVSwarmMPE(MultiAgentEnv):
    def __init__(
        self,
        num_team_a: int = 2,
        num_team_b: int = 2,
        num_landmarks: int = 3,
        # rewards / penalties
        local_collision_penalty: float = -1.0,
        spread_reward: float = 1.0,
        sum_over_landmarks: bool = True,
        tag_enemy_reward: float = 10.0,
        tag_friend_penalty: float = -10.0,
        fire_cost: float = 1.0,
        fire_radius: float = 0.2,
        fire_enemy_reward: float = 12.0,
        fire_friend_penalty: float = -12.0,
        move_cost_coef: float = 0.2,
        battery_capacity: float = 5.0,
        include_battery_in_obs: bool = True,
        vision_radius: Optional[float] = None,
        # core env params
        dim_p: int = 2,
        max_steps: int = MAX_STEPS,
        dt: float = DT,
        # physics overrides
        **kwargs,
    ):
        # sizes & names
        self.num_team_a = num_team_a
        self.num_team_b = num_team_b
        self.N = num_team_a + num_team_b
        self.L = num_landmarks
        self.num_entities = self.N + self.L

        self.team_a = [f"a_{i}" for i in range(num_team_a)]
        self.team_b = [f"b_{i}" for i in range(num_team_b)]
        self.agents = self.team_a + self.team_b
        self.landmarks = [f"landmark {i}" for i in range(num_landmarks)]
        self.a_to_i = {a: i for i, a in enumerate(self.agents)}

        # observation / action spaces (loose upper bound on obs length)
        obs_dim = (
            2 + 2                      # self vel, self pos
            + self.L * 2               # rel landmark pos
            + (self.N - 1) * 2 * 2     # rel agent pos & vel
            + 2                        # team one-hot
            + (1 if include_battery_in_obs else 0)
        )
        self.observation_spaces = {a: Box(-jnp.inf, jnp.inf, (obs_dim,)) for a in self.agents}
        self.action_spaces = {a: Box(-1.0, 1.0, (3,)) for a in self.agents}

        # env params
        self.dim_p = dim_p
        self.max_steps = max_steps
        self.dt = dt
        self.include_battery_in_obs = include_battery_in_obs
        self.vision_sq = None if vision_radius is None else vision_radius ** 2

        # physics params
        self.rad = kwargs.get(
            "rad",
            jnp.concatenate([jnp.full((self.N,), 0.06), jnp.full((self.L,), 0.05)])
        )
        self.moveable = kwargs.get(
            "moveable",
            jnp.concatenate([jnp.ones((self.N,), dtype=bool), jnp.zeros((self.L,), dtype=bool)])
        )
        self.mass = kwargs.get("mass", jnp.full((self.num_entities,), 1.0))
        self.accel = kwargs.get(
            "accel",
            jnp.concatenate([jnp.full((self.num_team_a,), 3.0),
                             jnp.full((self.num_team_b,), 4.0)])
        )
        self.max_speed = kwargs.get(
            "max_speed",
            jnp.concatenate([jnp.full((self.N,), 1.2), jnp.full((self.L,), 0.0)])
        )
        self.u_noise = kwargs.get("u_noise", jnp.zeros((self.N,)))
        self.damping = kwargs.get("damping", DAMPING)

        # reward / energy params
        self.local_collision_penalty = local_collision_penalty
        self.spread_reward = spread_reward
        self.sum_over_landmarks = sum_over_landmarks
        self.tag_enemy_reward = tag_enemy_reward
        self.tag_friend_penalty = tag_friend_penalty
        self.fire_cost = fire_cost
        self.fire_radius_sq = fire_radius ** 2
        self.fire_enemy_reward = fire_enemy_reward
        self.fire_friend_penalty = fire_friend_penalty
        self.move_cost_coef = move_cost_coef
        self.battery_capacity = battery_capacity

        # masks / constants
        self.team_mask_a = jnp.array([True] * num_team_a + [False] * num_team_b)
        self.team_mask_b = ~self.team_mask_a
        self.team_feat = jnp.stack(
            [self.team_mask_a.astype(jnp.float32), self.team_mask_b.astype(jnp.float32)], axis=-1
        )
        self.agent_range = jnp.arange(self.N)
        self.eye_N = jnp.eye(self.N, dtype=bool)
        self.rad_agents = self.rad[: self.N]

    # ------------------------------------------------------------------ #
    #                               Reset                                #
    # ------------------------------------------------------------------ #
    @partial(jax.jit, static_argnums=0)
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        key_a, key_l = jax.random.split(key)
        p_pos = jnp.concatenate(
            [
                jax.random.uniform(key_a, (self.N, 2), minval=-1.0, maxval=+1.0),
                jax.random.uniform(key_l, (self.L, 2), minval=-1.0, maxval=+1.0),
            ]
        )
        state = State(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_entities, self.dim_p)),
            done=jnp.zeros((self.N,), dtype=bool),
            step=0,
            active=jnp.ones((self.N,), dtype=bool),
            battery=jnp.full((self.N,), self.battery_capacity, dtype=jnp.float32),
        )
        return self.get_obs(state), state

    # ------------------------------------------------------------------ #
    #                                Step                                #
    # ------------------------------------------------------------------ #
    @partial(jax.jit, static_argnums=0)
    def step_env(self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]):
        raw = jnp.stack([actions[a] for a in self.agents], axis=0)  # [N,3]
        chex.assert_shape(raw, (self.N, 3))

        throttle = raw[:, :2]
        fire_bit = raw[:, 2] > 0.5

        # physics (agents only)
        u = throttle * self.accel[:, None]
        key, key_w = jax.random.split(key)
        p_pos, p_vel = self._world_step(key_w, state, u)

        # battery
        move_cost = self.move_cost_coef * jnp.sum(jnp.abs(throttle), axis=1)
        can_fire   = state.active & (state.battery >= (self.fire_cost + move_cost))
        effective_fire = fire_bit & can_fire
        total_cost = move_cost + self.fire_cost * effective_fire.astype(jnp.float32)
        new_battery = jnp.maximum(state.battery - total_cost, 0.0)

        # distances
        p_pos_a = p_pos[: self.N]
        diff = p_pos_a[:, None, :] - p_pos_a[None, :, :]
        dist_sq = jnp.sum(diff * diff, axis=-1)

        # collision detection (no forces)
        radii_sum = self.rad_agents[:, None] + self.rad_agents[None, :]
        coll_mat = (dist_sq < radii_sum ** 2) & (~self.eye_N)

        # tag: any enemy collision deactivates the victim
        enemy_pair = (self.team_mask_a[:, None] & self.team_mask_b[None, :]) | \
                     (self.team_mask_b[:, None] & self.team_mask_a[None, :])
        tag_hits_enemy = coll_mat & enemy_pair
        tag_victims = jnp.any(tag_hits_enemy.T, axis=1)

        # fire AoE: victim hit by >=2 shooters
        aoe_hits = (dist_sq <= self.fire_radius_sq) & (~self.eye_N)
        aoe_hits = aoe_hits & effective_fire[:, None] & state.active[None, :]
        shooters_per_victim = jnp.sum(aoe_hits, axis=0)
        fire_victims = shooters_per_victim >= 2

        # battery death
        battery_dead = new_battery <= 0.0

        victims = tag_victims | fire_victims | battery_dead
        new_active = state.active & (~victims)

        # termination
        team_a_dead = ~jnp.any(new_active[self.team_mask_a])
        team_b_dead = ~jnp.any(new_active[self.team_mask_b])
        done_all = team_a_dead | team_b_dead | ((state.step + 1) >= self.max_steps)

        per_agent_done = ~new_active
        done_vec = jnp.where(per_agent_done, True, done_all)

        next_state = State(
            p_pos=p_pos,
            p_vel=p_vel,
            done=done_vec,
            step=state.step + 1,
            active=new_active,
            battery=new_battery,
        )

        rewards_vec = self._rewards_vec(next_state, coll_mat, tag_hits_enemy, aoe_hits)

        obs = self.get_obs(next_state)
        dones = {a: done_vec[i] for i, a in enumerate(self.agents)}
        dones["__all__"] = done_all
        rewards = {a: rewards_vec[i] for i, a in enumerate(self.agents)}
        return obs, next_state, rewards, dones, {}

    # ------------------------------------------------------------------ #
    #                            Observations                             #
    # ------------------------------------------------------------------ #
    @partial(jax.jit, static_argnums=0)
    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        obs = self._obs_tensor(state)
        return {a: obs[i] for i, a in enumerate(self.agents)}

    def _obs_tensor(self, state: State) -> chex.Array:
        p_pos_a = state.p_pos[: self.N]
        p_vel_a = state.p_vel[: self.N]
        p_pos_l = state.p_pos[self.N:]

        active_f = state.active[:, None].astype(jnp.float32)
        p_pos_a = p_pos_a * active_f
        p_vel_a = p_vel_a * active_f

        rel_land = p_pos_l[None, :, :] - p_pos_a[:, None, :]            # [N, L, 2]
        rel_pos  = p_pos_a[None, :, :] - p_pos_a[:, None, :]            # [N, N, 2]
        rel_vel  = p_vel_a[None, :, :].repeat(self.N, axis=0)           # [N, N, 2]

        rel_pos_ns = rel_pos[~self.eye_N].reshape(self.N, self.N - 1, 2)
        rel_vel_ns = rel_vel[~self.eye_N].reshape(self.N, self.N - 1, 2)

        if self.vision_sq is not None:
            lm_mask = jnp.sum(rel_land ** 2, axis=-1) <= self.vision_sq
            ag_mask = jnp.sum(rel_pos_ns ** 2, axis=-1) <= self.vision_sq
            rel_land  = rel_land * lm_mask[..., None]
            rel_pos_ns = rel_pos_ns * ag_mask[..., None]
            rel_vel_ns = rel_vel_ns * ag_mask[..., None]

        parts = [
            p_vel_a,
            p_pos_a,
            rel_land.reshape(self.N, self.L * 2),
            rel_pos_ns.reshape(self.N, (self.N - 1) * 2),
            rel_vel_ns.reshape(self.N, (self.N - 1) * 2),
            self.team_feat,
        ]
        if self.include_battery_in_obs:
            parts.append(state.battery[:, None])

        return jnp.concatenate(parts, axis=-1)

    # ------------------------------------------------------------------ #
    #                               Reward                                #
    # ------------------------------------------------------------------ #
    def _rewards_vec(
        self,
        state: State,
        coll_mat: chex.Array,
        tag_hits_enemy: chex.Array,
        aoe_hits: chex.Array,
    ) -> chex.Array:
        active_f = state.active.astype(jnp.float32)

        # collision penalty
        act_pair = state.active[:, None] & state.active[None, :]
        col_pen = self.local_collision_penalty * jnp.sum(coll_mat & act_pair, axis=1).astype(jnp.float32)

        # spread reward
        p_pos_a = state.p_pos[: self.N]
        p_pos_l = state.p_pos[self.N:]
        al_diff = p_pos_a[:, None, :] - p_pos_l[None, :, :]
        al_dist_sq = jnp.sum(al_diff ** 2, axis=-1)
        if self.vision_sq is None:
            lm_mask = jnp.ones_like(al_dist_sq, dtype=bool)
        else:
            lm_mask = al_dist_sq <= self.vision_sq
        lm_mask = lm_mask & state.active[:, None]
        if self.sum_over_landmarks:
            spread_term = self.spread_reward * jnp.sum(lm_mask.astype(jnp.float32), axis=-1)
        else:
            spread_term = self.spread_reward * jnp.any(lm_mask, axis=-1).astype(jnp.float32)

        # tag term
        enemy_tags = jnp.sum(tag_hits_enemy, axis=1).astype(jnp.float32) * active_f
        friend_tags = jnp.sum((coll_mat & ~tag_hits_enemy), axis=1).astype(jnp.float32) * active_f
        tag_term = self.tag_enemy_reward * enemy_tags + self.tag_friend_penalty * friend_tags

        # fire term
        enemy_pair = (self.team_mask_a[:, None] & self.team_mask_b[None, :]) | \
                     (self.team_mask_b[:, None] & self.team_mask_a[None, :])
        fire_enemy = aoe_hits & enemy_pair
        fire_friend = aoe_hits & ~enemy_pair
        fire_enemy_counts = jnp.sum(fire_enemy, axis=1).astype(jnp.float32)
        fire_friend_counts = jnp.sum(fire_friend, axis=1).astype(jnp.float32)
        fire_term = (self.fire_enemy_reward * fire_enemy_counts +
                     self.fire_friend_penalty * fire_friend_counts) * active_f

        return spread_term + col_pen + tag_term + fire_term

    # ------------------------------------------------------------------ #
    #                               Physics                               #
    # ------------------------------------------------------------------ #
    def _world_step(self, key: chex.PRNGKey, state: State, u: chex.Array):
        """Apply action forces (with noise) and integrate. No env forces."""
        # noise
        keys = jax.random.split(key, self.N)
        noise = jax.vmap(lambda k, s: jax.random.normal(k, (2,)) * s)(keys, self.u_noise)
        p_force_agents = jnp.where(self.moveable[: self.N, None], u + noise, 0.0)

        # pad for landmarks
        p_force = jnp.concatenate([p_force_agents, jnp.zeros((self.L, 2))], axis=0)

        # integrate
        p_pos = state.p_pos + state.p_vel * self.dt
        p_vel = state.p_vel * (1 - self.damping) + (p_force / self.mass[:, None]) * self.dt * self.moveable[:, None]

        # clamp speed
        speed = jnp.linalg.norm(p_vel, axis=1)
        max_s = self.max_speed
        over = (speed > max_s) & (max_s >= 0)
        safe_speed = jnp.where(speed == 0, 1.0, speed)
        p_vel = jnp.where(over[:, None], p_vel / safe_speed[:, None] * max_s[:, None], p_vel)

        return p_pos, p_vel

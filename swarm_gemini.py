import jax
import jax.numpy as jnp
import chex
from flax import struct
from functools import partial
from typing import Dict, Tuple

from jaxmarl.environments.mpe.simple import SimpleMPE, State as BaseState
from jaxmarl.environments.spaces import Box
from jaxmarl.environments.mpe.default_params import *


@struct.dataclass
class State(BaseState):
    """Extended state for SwarmVSwarmMPE, including agent status."""
    active: chex.Array
    done: chex.Array
    battery: chex.Array


class SwarmVSwarmMPE(SimpleMPE):
    """
    Homogeneous two-swarm MPE with:
      • Limited-vision observations
      • Individual rewards (spread / tag / fire)
      • Tag or ≥2-fire AoE deactivates agents
      • Battery system: fire has fixed cost, movement costs ∝ throttle
      • Agents deactivate (and are done) at battery <= 0
      • Episode ends on team wipe or max_steps
      • Tensor-first APIs (+ dict wrappers)
    """

    def __init__(
        self,
        num_team_a: int = 2,
        num_team_b: int = 2,
        num_landmarks: int = 3,
        tag_ratio: float = 0.5,
        local_collision_penalty: float = -1.0,
        spread_reward: float = 1.0,
        sum_over_landmarks: bool = True,
        # Tag rewards
        tag_enemy_reward: float = 10.0,
        tag_friend_penalty: float = -10.0,
        # Fire
        fire_cost: float = 1.0,
        fire_radius: float = 0.2,
        fire_enemy_reward: float = 12.0,
        fire_friend_penalty: float = -12.0,
        # Movement energy
        move_cost_coef: float = 0.2,
        battery_capacity: float = 5.0,
        include_battery_in_obs: bool = True,
        # Vision
        vision_radius: float | None = None,
        action_type=CONTINUOUS_ACT,
        **kwargs,
    ):
        assert 0.0 <= tag_ratio <= 1.0
        self.num_team_a = num_team_a
        self.num_team_b = num_team_b
        self.N = num_team_a + num_team_b
        self.L = num_landmarks

        # Rewards / shaping
        self.tag_ratio = tag_ratio
        self.local_collision_penalty = local_collision_penalty
        self.spread_reward = spread_reward
        self.sum_over_landmarks = sum_over_landmarks

        self.tag_enemy_reward = tag_enemy_reward
        self.tag_friend_penalty = tag_friend_penalty

        self.fire_cost = fire_cost
        self.fire_radius = fire_radius
        self.fire_radius_sq = fire_radius**2
        self.fire_enemy_reward = fire_enemy_reward
        self.fire_friend_penalty = fire_friend_penalty

        self.move_cost_coef = move_cost_coef
        self.battery_capacity = battery_capacity
        self.include_battery_in_obs = include_battery_in_obs

        self.vision_radius = vision_radius
        self.vision_sq = None if vision_radius is None else float(vision_radius**2)

        # Names
        self.team_a = [f"a_{i}" for i in range(num_team_a)]
        self.team_b = [f"b_{i}" for i in range(num_team_b)]
        agents = self.team_a + self.team_b
        landmarks = [f"landmark {i}" for i in range(num_landmarks)]

        # Team feats/masks
        self.team_feat = jnp.concatenate(
            [jnp.tile(jnp.array([1.0, 0.0]), (num_team_a, 1)),
             jnp.tile(jnp.array([0.0, 1.0]), (num_team_b, 1))],
            axis=0,
        )
        self.team_mask_a = jnp.concatenate(
            [jnp.ones(num_team_a, dtype=bool), jnp.zeros(num_team_b, dtype=bool)]
        )
        self.team_mask_b = ~self.team_mask_a

        team_feat_dim = self.team_feat.shape[1]
        self.obs_dim = (
            2 + 2
            + num_landmarks * 2
            + (self.N - 1) * 2
            + (self.N - 1) * 2
            + team_feat_dim
            + (1 if include_battery_in_obs else 0)
        )
        observation_spaces = {a: Box(-jnp.inf, jnp.inf, (self.obs_dim,)) for a in agents}

        # Physics params
        colour = (
            [ADVERSARY_COLOUR] * num_team_a
            + [AGENT_COLOUR] * num_team_b
            + [OBS_COLOUR] * num_landmarks
        )
        rad = jnp.concatenate(
            [jnp.full((num_team_a), 0.075),
             jnp.full((num_team_b), 0.05),
             jnp.full((num_landmarks), 0.05)]
        )
        accel = jnp.concatenate(
            [jnp.full((num_team_a), 3.0),
             jnp.full((num_team_b), 4.0)]
        )
        max_speed = jnp.concatenate(
            [jnp.full((num_team_a), 1.0),
             jnp.full((num_team_b), 1.3),
             jnp.full((num_landmarks), 0.0)]
        )
        collide = jnp.concatenate(
            [jnp.full((self.N), True), jnp.full((num_landmarks), False)]
        )

        super().__init__(
            num_agents=self.N,
            agents=agents,
            num_landmarks=num_landmarks,
            landmarks=landmarks,
            action_type=action_type,
            observation_spaces=observation_spaces,
            dim_c=0,
            colour=colour,
            rad=rad,
            accel=accel,
            max_speed=max_speed,
            collide=collide,
            **kwargs,
        )

        self.eye_N = jnp.eye(self.N, dtype=bool)
        self.rad_agents = self.rad[: self.N]

    # ---------- Helpers ----------
    @staticmethod
    def _ensure_batch(x: chex.Array) -> Tuple[chex.Array, bool]:
        if x.ndim == 2:
            return x[None, ...], False
        return x, True

    # ---------- Reset ----------
    def reset(self, key: chex.PRNGKey) -> State:
        base_state = super().reset(key)
        active = jnp.ones((self.N,), dtype=bool)
        done = jnp.zeros((self.N,), dtype=bool)
        battery = jnp.full((self.N,), self.battery_capacity, dtype=jnp.float32)
        return State(**base_state.__dict__, active=active, done=done, battery=battery)

    # ---------- Step ----------
    @partial(jax.jit, static_argnums=0)
    def step_env(self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]):
        # actions[a] = [ax, ay, fire_bit]
        act_arr = jnp.stack([actions[a] for a in self.agents], axis=0)  # [N,3]
        chex.assert_shape(act_arr, (self.N, 3))
        ax_ay = act_arr[:, :2]
        fire_bit = act_arr[:, 2] > 0.5

        move_cost = self.move_cost_coef * jnp.sum(jnp.abs(ax_ay), axis=1) # [cite: 6]
        can_fire = state.active & (state.battery >= (self.fire_cost + move_cost))
        effective_fire = fire_bit & can_fire # [cite: 5]

        # (Optional) gate movement as well; here we allow movement even if it drains to 0
        ax_ay = ax_ay * state.active[:, None].astype(ax_ay.dtype)

        # The base SimpleMPE step_env expects a dict of continuous actions [ax, ay]
        # It does not handle the communication/fire part of the action space
        # We create a dummy actions dict just for the movement part
        move_actions = {a: ax_ay[i] for i, a in enumerate(self.agents)}

        # Parent physics
        next_state_base, _, _, _ = super().step_env(key, state, move_actions)

        # Battery bookkeeping
        total_cost = move_cost + self.fire_cost * effective_fire.astype(jnp.float32)
        new_battery = jnp.maximum(state.battery - total_cost, 0.0)

        # Collisions (tag)
        p_pos_agents = next_state_base.p_pos[: self.N]
        diff = p_pos_agents[:, None, :] - p_pos_agents[None, :, :]
        dist_sq = jnp.sum(diff * diff, axis=-1)
        r_sum = self.rad_agents[:, None] + self.rad_agents[None, :]
        coll_mat = (dist_sq < (r_sum**2)) & (~self.eye_N)

        tag_hits_enemy = coll_mat & (
            (self.team_mask_a[:, None] & self.team_mask_b[None, :])
            | (self.team_mask_b[:, None] & self.team_mask_a[None, :])
        )
        tag_victims = jnp.any(tag_hits_enemy.T, axis=1) # [cite: 8]

        # Fire AoE
        aoe_hits = (dist_sq <= self.fire_radius_sq) & (~self.eye_N)
        aoe_hits = aoe_hits & effective_fire[:, None] & state.active[None, :]
        shooters_per_victim = jnp.sum(aoe_hits, axis=0)
        fire_victims = shooters_per_victim >= 2 # [cite: 9]

        # Battery death
        battery_dead = new_battery <= 0.0 # [cite: 7]

        victims = tag_victims | fire_victims | battery_dead # [cite: 12]
        new_active = state.active & (~victims)

        # Termination
        team_a_dead = ~jnp.any(new_active[self.team_mask_a])
        team_b_dead = ~jnp.any(new_active[self.team_mask_b])
        done_all = (team_a_dead | team_b_dead) | (next_state_base.step >= self.max_steps) # [cite: 12]

        per_agent_done = ~new_active # [cite: 13]
        done_vec = jnp.where(per_agent_done, True, done_all)

        next_state = State(
            **next_state_base.__dict__,
            active=new_active,
            done=done_vec,
            battery=new_battery
        )

        rewards = self._rewards_vec(next_state, coll_mat, tag_hits_enemy, aoe_hits)
        obs = self.get_obs(next_state)

        dones = {a: done_vec[i] for i, a in enumerate(self.agents)}
        dones["__all__"] = done_all # [cite: 13]
        
        # Convert rewards vec to dict
        rewards_dict = {a: rewards[i] for i, a in enumerate(self.agents)}
        return obs, next_state, rewards_dict, dones, {}


    # ---------- OBS tensor + wrapper ----------
    def get_obs_tensor(self, state: State) -> chex.Array:
        p_pos, batched = self._ensure_batch(state.p_pos)
        p_vel, _ = self._ensure_batch(state.p_vel)
        active, _ = self._ensure_batch(state.active.astype(jnp.float32))
        battery, _ = self._ensure_batch(state.battery.astype(jnp.float32))

        p_pos_agents = p_pos[:, : self.N]
        p_vel_agents = p_vel[:, : self.N]
        p_pos_lm = p_pos[:, self.N :]

        p_pos_agents = p_pos_agents * active[..., None]
        p_vel_agents = p_vel_agents * active[..., None]

        # Relative positions
        rel_land = p_pos_lm[:, None, :, :] - p_pos_agents[:, :, None, :]
        rel_pos = p_pos_agents[:, None, :, :] - p_pos_agents[:, :, None, :]
        
        # Relative velocities
        rel_vel = p_vel_agents[:, None, :, :] - p_vel_agents[:, :, None, :]

        # Remove self-observation
        rel_pos_ns = rel_pos[:, ~self.eye_N].reshape(-1, self.N, self.N - 1, 2)
        rel_vel_ns = rel_vel[:, ~self.eye_N].reshape(-1, self.N, self.N - 1, 2)

        # Limited vision
        if self.vision_sq is not None:
            land_d2 = jnp.sum(rel_land**2, axis=-1)
            agent_d2 = jnp.sum(rel_pos_ns**2, axis=-1)
            lm_mask = land_d2 <= self.vision_sq
            ag_mask = agent_d2 <= self.vision_sq
            rel_land   = rel_land * lm_mask[..., None]
            rel_pos_ns = rel_pos_ns * ag_mask[..., None]
            rel_vel_ns = rel_vel_ns * ag_mask[..., None]

        B = p_pos.shape[0]
        team_feat_b = jnp.broadcast_to(self.team_feat, (B, self.N, self.team_feat.shape[1]))

        parts = [
            p_vel_agents, # self vel
            p_pos_agents, # self pos
            rel_land.reshape(B, self.N, self.L * 2),
            rel_pos_ns.reshape(B, self.N, (self.N - 1) * 2),
            rel_vel_ns.reshape(B, self.N, (self.N - 1) * 2),
            team_feat_b,
        ]
        if self.include_battery_in_obs:
            parts.append(battery[..., None])

        obs = jnp.concatenate(parts, axis=-1)
        return obs if batched else obs[0]

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        obs = self.get_obs_tensor(state)
        if obs.ndim == 2:
            return {a: obs[i] for i, a in enumerate(self.agents)}
        return {a: obs[:, i] for i, a in enumerate(self.agents)}

    # ---------- Reward vector (single env) ----------
    def _rewards_vec(
        self,
        state: State,
        coll_mat: chex.Array,
        tag_hits_enemy: chex.Array,
        aoe_hits: chex.Array,
    ) -> chex.Array:
        active = state.active

        # Mask pairs by active for penalties
        act_pair = active[:, None] & active[None, :]
        spread_collision_pen = self.local_collision_penalty * jnp.sum(coll_mat & act_pair, axis=1).astype(jnp.float32) # [cite: 11]

        # Spread
        p_pos_agents = state.p_pos[: self.N]
        p_pos_lm = state.p_pos[self.N :]
        al_diff_sq = p_pos_agents[:, None, :] - p_pos_lm[None, :, :]
        al_dist_sq = jnp.sum(al_diff_sq * al_diff_sq, axis=-1)
        if self.vision_sq is None:
            lm_mask = jnp.ones_like(al_dist_sq, dtype=bool)
        else:
            lm_mask = al_dist_sq <= self.vision_sq
        lm_mask = lm_mask & active[:, None]

        if self.sum_over_landmarks:
            spread_term = self.spread_reward * jnp.sum(lm_mask.astype(jnp.float32), axis=-1)
        else:
            spread_term = self.spread_reward * jnp.any(lm_mask, axis=-1).astype(jnp.float32)

        # Tag term
        enemy_tags = jnp.sum(tag_hits_enemy, axis=1).astype(jnp.float32)
        friend_tags = jnp.sum((coll_mat & ~tag_hits_enemy), axis=1).astype(jnp.float32)
        enemy_tags *= active.astype(jnp.float32)
        friend_tags *= active.astype(jnp.float32)
        tag_term = self.tag_enemy_reward * enemy_tags + self.tag_friend_penalty * friend_tags # [cite: 8]

        # Fire term
        fire_enemy = aoe_hits & (
            (self.team_mask_a[:, None] & self.team_mask_b[None, :])
            | (self.team_mask_b[:, None] & self.team_mask_a[None, :])
        )
        fire_friend = aoe_hits & ~fire_enemy
        fire_enemy_counts = jnp.sum(fire_enemy, axis=1).astype(jnp.float32)
        fire_friend_counts = jnp.sum(fire_friend, axis=1).astype(jnp.float32)
        fire_term = (self.fire_enemy_reward * fire_enemy_counts +
                     self.fire_friend_penalty * fire_friend_counts) * active.astype(jnp.float32) # [cite: 10]

        per_agent_spread = spread_term + spread_collision_pen
        final = (1.0 - self.tag_ratio) * per_agent_spread + self.tag_ratio * tag_term + fire_term
        return final

    # Dict wrapper
    def rewards(self, state: State) -> Dict[str, chex.Array]:
        """
        Calculates rewards for the given state.
        
        NOTE: This wrapper cannot calculate fire rewards as it does not have access 
        to agent actions from the step. Fire-related rewards will be zero when 
        calling this method directly. The main `step_env` loop calculates all
        rewards correctly.
        """
        p_pos_agents = state.p_pos[: self.N]
        diff = p_pos_agents[:, None, :] - p_pos_agents[None, :, :]
        dist_sq = jnp.sum(diff * diff, axis=-1)
        r_sum = self.rad_agents[:, None] + self.rad_agents[None, :]
        coll_mat = (dist_sq < (r_sum**2)) & (~self.eye_N)
        tag_hits_enemy = coll_mat & (
            (self.team_mask_a[:, None] & self.team_mask_b[None, :])
            | (self.team_mask_b[:, None] & self.team_mask_a[None, :])
        )
        # Fire rewards cannot be calculated without actions, so aoe_hits is all False.
        aoe_hits = jnp.zeros_like(coll_mat, dtype=bool)

        vec = self._rewards_vec(state, coll_mat, tag_hits_enemy, aoe_hits)
        return {a: vec[i] for i, a in enumerate(self.agents)}

import jax
import jax.numpy as jnp
import chex
from typing import Dict
from functools import partial
from jaxmarl.environments.mpe.simple import SimpleMPE, State
from jaxmarl.environments.mpe.default_params import *
from jaxmarl.environments.spaces import Box


class SimpleSpreadMPE(SimpleMPE):
    """Two-team Simple Spread, no comms, local observations within view_radius."""
    def __init__(
        self,
        num_agents: int = 6,
        num_landmarks: int = 3,
        local_ratio: float = 0.5,
        view_radius: float = 1.0,
        action_type=DISCRETE_ACT,
        **kwargs,
    ):
        # --- fixed two-team split (first half team 0, rest team 1) ---
        half = num_agents // 2
        self.team_ids = jnp.concatenate(
            [jnp.zeros((half,), dtype=jnp.int32),
             jnp.ones((num_agents - half,), dtype=jnp.int32)]
        )

        self.view_radius = view_radius

        agents = [f"agent_{i}" for i in range(num_agents)]
        landmarks = [f"landmark {i}" for i in range(num_landmarks)]

        # obs: vel2 + pos2 + 2*L + 2*(N-1) + (N-1)same-team flags
        obs_dim = 4 + 2 * num_landmarks + 2 * (num_agents - 1) + (num_agents - 1)
        observation_spaces = {a: Box(-jnp.inf, jnp.inf, (obs_dim,)) for a in agents}

        colour = [AGENT_COLOUR] * num_agents + [OBS_COLOUR] * num_landmarks

        self.local_ratio = local_ratio
        assert 0.0 <= self.local_ratio <= 1.0

        rad = jnp.concatenate(
            [jnp.full((num_agents,), 0.15), jnp.full((num_landmarks,), 0.05)]
        )
        collide = jnp.concatenate(
            [jnp.full((num_agents,), True), jnp.full((num_landmarks,), False)]
        )

        super().__init__(
            num_agents=num_agents,
            agents=agents,
            num_landmarks=num_landmarks,
            landmarks=landmarks,
            action_type=action_type,
            observation_spaces=observation_spaces,
            colour=colour,
            rad=rad,
            collide=collide,
            dim_c=0,
            **kwargs,
        )

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Mask all entities beyond view_radius by zeroing their relative features."""
        team_ids = self.team_ids
        vr2 = self.view_radius ** 2  # compare squared distances (no sqrt)

        @partial(jax.vmap, in_axes=(0,))
        def _gather(aidx: int):
            # Rel positions
            rel_land = state.p_pos[self.num_agents:] - state.p_pos[aidx]       # [L,2]
            rel_others = state.p_pos[: self.num_agents] - state.p_pos[aidx]    # [N,2]

            # drop ego
            rel_others = jnp.roll(rel_others, shift=self.num_agents - aidx - 1, axis=0)[: self.num_agents - 1]
            rel_others = jnp.roll(rel_others, shift=aidx, axis=0)

            # distance masks (squared)
            land_mask = (jnp.sum(rel_land ** 2, axis=1) <= vr2).astype(jnp.float32)     # [L]
            other_mask = (jnp.sum(rel_others ** 2, axis=1) <= vr2).astype(jnp.float32)  # [N-1]

            # apply masks
            rel_land = rel_land * land_mask[:, None]
            rel_others = rel_others * other_mask[:, None]

            # same-team flags (still shown for all others; optionally mask with visibility)
            other_team_ids = jnp.roll(team_ids, shift=self.num_agents - aidx - 1)[: self.num_agents - 1]
            other_team_ids = jnp.roll(other_team_ids, shift=aidx, axis=0)
            same_team_flags = (other_team_ids == team_ids[aidx]).astype(jnp.float32)

            # if you want invisibles to have 0 team flag, uncomment next line:
            # same_team_flags = same_team_flags * other_mask

            return rel_land, rel_others, same_team_flags, land_mask, other_mask

        rel_land, rel_others, same_team_flags, _, _ = _gather(self.agent_range)

        def _obs(i: int):
            return jnp.concatenate(
                [
                    state.p_vel[i],                    # 2
                    state.p_pos[i],                    # 2
                    rel_land[i].flatten(),             # 2*L
                    rel_others[i].flatten(),           # 2*(N-1)
                    same_team_flags[i].flatten(),      # (N-1)
                ]
            )

        return {a: _obs(i) for i, a in enumerate(self.agents)}

    def rewards(self, state: State) -> Dict[str, float]:
        # unchanged vanilla spread reward
        @partial(jax.vmap, in_axes=(0, None))
        def _collisions(agent_idx: int, other_idx: int):
            return jax.vmap(self.is_collision, in_axes=(None, 0, None))(
                agent_idx, other_idx, state
            )

        collisions = _collisions(self.agent_range, self.agent_range)
        local_pen = -jnp.sum(collisions, axis=1)

        def _land(land_pos):
            d = state.p_pos[: self.num_agents] - land_pos
            return -jnp.min(jnp.linalg.norm(d, axis=1))

        global_rew = jnp.sum(jax.vmap(_land)(state.p_pos[self.num_agents:]))

        return {
            a: local_pen[i] * self.local_ratio + global_rew * (1.0 - self.local_ratio)
            for i, a in enumerate(self.agents)
        }

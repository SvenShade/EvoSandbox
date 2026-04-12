# INFO --------------------------------------------------------------------- #

# Author:  Steven Spratley
#
# Purpose: A 3D swarm vs. swarm scenario involving multiple objectives.
#          Extends JaxMARL's implementation of simple_spread, from the
#          Multi-agent Particle Environment (MPE).
#
# Scenario rules:
#  - Agents spawn in two even teams, above randomised terrain.
#  - Landmarks are flat regions of interest, and spawn at ground level.
#  - Agents can accelerate in xyz
#  - Autofire controls firing behaviour.
#  - Actions have associated battery power costs.
#  - Hits received by neighbouring agents will deplete battery power.
#  - Agent deactivation is caused by collision (with land, other agents, 
#    or environment boundaries), and/or a fully depleted battery.
#  - Gravity affects all non-grounded agents.

# IMPORTS ------------------------------------------------------------------ #

import jax
import jax.numpy as jnp
import chex
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import pyvista as pv
from jax import lax
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Optional, Dict
from functools import partial
from flax import struct
from jaxmarl.environments.mpe.default_params import *
from jaxmarl.environments.spaces import Box
from jaxmarl.environments.multi_agent_env import MultiAgentEnv

# CONSTANTS AND GLOBAL SETTINGS -------------------------------------------- #

# Colours.
AGNT_CLR = 'dodgerblue'   # agent team 1
ADVR_CLR = 'orangered'    # agent team 2
LDMK_CLR = 'darkseagreen' # landmarks
LAND_CLR = 'dimgrey'      # terrain
DONE_CLR = 'lightgrey'    # deactivated agents
FIRE_CLR = 'yellow'       # agents who fired
PGON_CLR = 'darkviolet'   # clay pigeons

SEED = 42
RENDER_H = 1080
pv.global_theme.volume_mapper = 'gpu'

# ENVIRONMENT
ONE_SWARM = False # Whether agents form one or two teams
KILL_BOX  = False # Whether to remove terrain/obstacles and keep agents only
ATTITUDES = True  # Whether to use attitudes
PRFCT_ATT = False # Whether drones have perfect knowledge of operator att
ATT_ACTNS = True  # Whether non-leader drones can act to update their attitudes
HIDE_SEEK = False # Whether combat is asymmetric (only team zero can fire)
HALF_PRES = True  # Train with bfloat16 instead of float32
ADV_INACT = False # Freeze adversaries. A temporary/inefficient solution
                  # to benchmarking one-sided pred-prey.

# RENDERER
CLRMP_ATT = True  # Whether to map attitudes to rendered colours
DIFF_GEOM = True  # Whether to assign individual drone geometries to teams

PRE = jnp.bfloat16 if HALF_PRES else jnp.float32

# CLASSES ------------------------------------------------------------------ #

@struct.dataclass
class State:
    p_pos:     chex.Array # float [N, dim_p]
    p_vel:     chex.Array # float [N, dim_p]
    p_ori:     chex.Array # float [N, dim_p]
    pg_pos:    chex.Array # float [P, dim_p]
    pg_vel:    chex.Array # float [P, dim_p]
    pg_batt:   chex.Array # float [P]
    pg_done:   chex.Array # bool  [P]
    pos_hist:  chex.Array # float [N, hist_q_len, dim_p]
    vel_hist:  chex.Array # float [N, ori_hist, dim_p]
    actions:   chex.Array # float [N, dim_p + dim_a]
    attitudes: chex.Array # float [N, dim_a]
    op_att:    chex.Array # float [2, dim_a]
    speed:     chex.Array # float [N]
    batt:      chex.Array # float [N]
    fire:      chex.Array # bool  [N]
    fire_ago:  chex.Array # int   [N]
    hmap:      chex.Array # float [2, size, size]: small sample of full hmap
    last_expl_map: chex.Array # float [2, expl_grid_sz, expl_grid_sz]
    expl_map:  chex.Array     # float [2, expl_grid_sz, expl_grid_sz]
    snapshots: chex.Array # float [N, view_width, view_width]
    step:      int
    tot_steps: int
    warmup:    float
    warmup_multi: float
    transition: bool # whether state is used for a transitional render frame
    done:      chex.Array # bool  [N]
    done_ago:  chex.Array # int   [N]
    r_hits:    chex.Array # int   [N]: hits received by agent this step
    g_hits:    chex.Array # int   [N]: all hits landed by agent this step
    g_kills:   chex.Array # int   [N]: kills made by agent this step

class SimpleSpreadMPE(MultiAgentEnv):
    def __init__(
        self,
        num_agents,
        num_landmarks,
        view_frac,
        body_frac,
        env_scale,
        max_steps,
        ter_res,
        ter_oct,
        rew_scaling,
        warmup_on,
        warmup_steps,
        eval_mode,
        **kwargs,
    ):
        super().__init__(num_agents=num_agents)

        # Environment parameters.
        self.env_size     = int(50.0 * env_scale) # Environment size
        self.dim_p        = 3    # Spatial  dimensions
        self.dim_a        = 3    # Attitude dimensions
        self.height_lim   = 1.0 if KILL_BOX else 0.4 # Fraction of env size to limit z axis
        self.warmup_on    = warmup_on # Whether to use warmup phase
        self.max_steps    = max_steps
        self.hist_q_len   = 75   # Num positions to keep in history queue
        self.expl_decay   = 0.98 # Decay term for exploration grid
        self.dirich_alpha = 0.01 # 0.3  # Alpha for Dirichlet (used in sampling operator preferences)
        self.init_battery = 100.0
        self.throt_batt   = self.init_battery / 3000
        self.fire_batt    = self.init_battery / 10
        self.hit_batt     = self.init_battery / 2.5
        self.damping      = 0.05
        self.u_noise      = 0.0
        self.dt           = 0.1
        self.mass         = 1.0
        self.accel        = 10.0
        self.att_update   = 0.1
        self.gravity      = 0.1
        self.max_speed    = 15.0
        self.body_rad     = (body_frac * self.env_size ** 2 / math.pi) ** 0.5
        self.body_rad_sqr = jnp.square(self.body_rad)
        self.nrm_body_rad = self.body_rad / self.env_size
        self.expl_grid_sz = self.body_rad * 4 # Granularity of exploration grid
        self.eps          = 0.008 if HALF_PRES else 1e-7 # Small epsilon to avoid NaN
        self.death_anim   = 20 # Num timesteps in death animation
        self.ori_hist     = 5  # Num timesteps to average over when setting orientation
                               # This ensures craft can't turn around instantly
        self.bounds       = (0.0, self.env_size) * 3

        # Entity parameters.
        self.num_agnt     = num_agents
        self.num_ldmk     = num_landmarks
        self.num_pgon     = int(kwargs.get("num_pigeons", 0))
        self.num_entities = num_agents + num_landmarks + self.num_pgon
        self.default_att  = 0.5 # Default attitude setting, per self.dim_a
        self.init_att     = jnp.full((self.num_agnt, self.dim_a), self.default_att, dtype=PRE)
        self.view_rad     = (view_frac * self.env_size ** 2 / math.pi) ** 0.5
        self.view_rad_sqr = jnp.square(self.view_rad)
        self.view_rad_int = int(jnp.ceil(self.view_rad))
        self.view_width   = 9 # Must be odd
        self.fire_rad     = 0.5 * self.view_rad
        self.fire_rad_sqr = jnp.square(self.fire_rad)
        self.fire_cos     = 0.75 # Angle of fire cone
        self.fire_refr    = 5    # Fire refractory period
        self.obs_slots    = 5
        self.pgon_obs_slots = self.obs_slots
        self.pgon_init_batt = 2.0
        self.pgon_accel     = self.accel * 0.75
        self.pgon_max_speed = self.max_speed * 0.9
        self.pgon_safe_alt  = self.body_rad * 2.0
        self.pgon_terr_gain = self.pgon_accel * 1.25
        self.pgon_edge_gain = self.pgon_accel * 0.5
        _pi, _pj          = np.triu_indices(self.obs_slots, k=1)
        self.pair_i       = jnp.array(_pi, dtype=jnp.int32) # Used for pairwise neighbour geometry.
        self.pair_j       = jnp.array(_pj, dtype=jnp.int32)
        self.agents       = [f"agent_{i}" for i in range(num_agents)]
        assert not num_agents % 2 # Must be even for two teams
        # Observations always reserve self.obs_slots neighbour slots and zero-pad
        # when fewer entities exist or are visible.
        self.ts           = num_agents // 2
        if ONE_SWARM:
            self.team_ids = jnp.zeros(num_agents, dtype=jnp.int16)
        else:
            self.team_ids = jnp.concatenate(
                [jnp.zeros(self.ts, dtype=jnp.int16),
                 jnp.ones( self.ts, dtype=jnp.int16)]
            )
        self.team_0_idxs  = jnp.argwhere(1 - self.team_ids).flatten()
        self.team_1_idxs  = jnp.argwhere(    self.team_ids).flatten()
        self.frozen       = self.team_ids.astype(jnp.bool) if ADV_INACT \
                            else jnp.full(num_agents, False)

        # Reward multipliers.
        # Note: due to rewards being limited to a min of dest_rew per timestep ('death clipping'),
        # dest_rew must remain the largest reward multiplier.
        self.dest_rew     = 2     # if self is destroyed
        self.cmbt_rew     = 1     # combat
        self.cohe_rew     = 0.1   # distance to neighbours (cohesion)
        self.oper_rew     = 0.05  # similarity to operator intent
        self.sctg_rew     = 0.05  # scouting
        self.occp_rew     = 0.05  # occupation
        self.sped_rew     = 0.01  # speed
        self.cons_rew     = 0.005 # battery conservation

        # Other reward shaping variables.
        self.eval_mode    = eval_mode # whether to use eval reward function
        self.warmup_steps = warmup_steps # num timesteps to 'warm up' / perform curriculum
        self.safe_dist    = 0.5 # fraction of view_rad within which to cut cohesion rewards
        self.rew_scaling  = rew_scaling # whether attitudes scale reward terms

        # Action space [-1, 1]:
        #   Per-axis throttle
        #   Attitude update
        #       - combat     (red)
        #       - scouting   (blue)
        #       - occupation (green)
        self.action_spaces = {i: Box(-1.0, 1.0, (self.dim_p + self.dim_a,))
                                 for i in self.agents}

        # Observation space [-1, 1]:
        #   Ego: self pos, vel, attitude, fire, team, battery, last action
        #   Neighbours: pos, vel, attitude, team (all relative to ego), fire, invariants
        #   Clay pigeons: pos, vel, battery, invariants
        #   Neighbour pair geometry
        #   Terrain snapshot image
        ego_dim     =  self.dim_p*3 + self.dim_a*2 + 4
        nbor_dim    = (self.dim_p*2 + self.dim_a + 2 + 4) * self.obs_slots
        pgon_dim    = (self.dim_p*2 + 1 + 4) * self.obs_slots
        pair_dim    = self.obs_slots * (self.obs_slots - 1) // 2
        terrain_dim =  self.view_width ** 2
        self.observation_spaces = {
            a: Box(-10, 10, (ego_dim + nbor_dim + pgon_dim + pair_dim + terrain_dim,))
               for a in self.agents}

        # Initialise large terrain with areas of interest, 
        # to be sampled by training episodes.
        self.hmap_size = self.env_size * 32
        self.num_poi = int(self.num_ldmk * (self.hmap_size / self.env_size) ** 2)
        self.hmap = fractal2d(jax.random.key(SEED), 
                        size=self.hmap_size,
                        res=int(self.hmap_size / ter_res),
                        octaves=ter_oct,
                        clip_max=self.height_lim - 4 * self.nrm_body_rad,
                        num_poi=self.num_poi,
                        poi_rad=self.view_rad)
        if KILL_BOX:
            self.hmap = self.hmap * 0.0
    
    # HELPERS -------------------------------------------------------------- #

    def agent_pigeon_interactions(self, agnt_pos, agnt_done, pg_pos, pg_done):
        """Visible, valid agent->pigeon relative geometry."""
        diff_pos = pg_pos[None, :, :] - agnt_pos[:, None, :]
        sqr_dist = jnp.sum(diff_pos ** 2, axis=-1)
        keep = (sqr_dist < self.view_rad_sqr) & ((~agnt_done)[:, None] & (~pg_done)[None, :])
        diff_pos = jnp.where(keep[..., None], diff_pos, 0.0)
        sqr_dist = jnp.where(keep, sqr_dist, 0.0)
        return diff_pos, sqr_dist

    def pigeon_agent_collisions(self, agnt_pos, agnt_done, pg_pos, pg_done):
        if self.num_pgon == 0:
            return jnp.zeros((0,), dtype=jnp.int32)
        diff_pos = pg_pos[None, :, :] - agnt_pos[:, None, :]
        sqr_dist = jnp.sum(diff_pos ** 2, axis=-1)
        coll = ((~agnt_done)[:, None] & (~pg_done)[None, :]
                & (sqr_dist > 0.0)
                & (sqr_dist < self.body_rad_sqr * 2))
        return jnp.sum(coll, axis=0).astype(jnp.int32)

    # Compute agent interactions used in step_env, reset, obs, and rewards.
    # Returns relative positions, squared distances, and collisions, masked
    # to contain positive floats for valid and visible interactions only.
    def interactions(self, state: State):
        # Calculate relative positions and squared dists between all agents.
        diff_pos = state.p_pos[None, :] - state.p_pos[:, None] # [N,N,dim_p]
        sqr_dist = jnp.sum(diff_pos ** 2, axis=-1)
        # Mask by visibility, i.e. agents within view_rad of each other.
        visib_mask = (sqr_dist < self.view_rad_sqr).astype(PRE)
        diff_pos = diff_pos * visib_mask[..., None]
        sqr_dist = sqr_dist * visib_mask
        # Mask by valid interactions, i.e pairs of active agents.
        valid_mask = ~((state.done[None, :] | state.done[:, None]) 
                       | jnp.eye(self.num_agnt, dtype=jnp.bool))
        diff_pos = diff_pos * valid_mask.astype(PRE)[..., None]
        sqr_dist = sqr_dist * valid_mask.astype(PRE)

        # Calculate relative positions and squared dists from agents to clay pigeons.
        pg_diff_pos, pg_sqr_dist = self.agent_pigeon_interactions(
            state.p_pos,
            state.done,
            state.pg_pos,
            state.pg_done,
        )

        # Find agent-agent and agent-terrain collisions.
        agnt_colls   = jnp.sum((sqr_dist < self.body_rad_sqr * 2) 
                             & (sqr_dist > 0), axis=-1) # int [N]
        norm_pos     = state.p_pos / self.env_size
        land_heights = sample_hmap(state.hmap, norm_pos[:, :-1])
        land_colls   = (norm_pos[:, -1] - land_heights) < self.nrm_body_rad

        # Clay-pigeon terrain heights are used by the flee controller.
        pg_land_heights = sample_hmap(state.hmap, state.pg_pos[:, :2] / self.env_size)

        # Create aerial 'snapshots' for agent observations.
        aerials   = self.observe_hmap(state)
        out_shape = (self.num_agnt, self.view_width, self.view_width)
        snapshots = jax.image.resize((aerials + 1) / 2, out_shape, "bilinear").astype(PRE)
        return (
            diff_pos,
            sqr_dist,
            pg_diff_pos,
            pg_sqr_dist,
            agnt_colls,
            land_heights,
            land_colls,
            pg_land_heights,
            aerials,
            snapshots,
        )
    
    # Zoom in on a region of self.hmap to define current episode's terrain.
    def index_hmap(self, key: chex.PRNGKey):
        key, sub = jax.random.split(key)
        idxs = jax.random.randint(
            sub, (2,), minval=0, maxval=self.hmap_size - self.env_size + 1)
        return lax.dynamic_slice(self.hmap, start_indices=[0, *idxs], 
            slice_sizes=(2, self.env_size, self.env_size))
    
    # Perform nearest-neighbour sampling on an array, given agent positions
    # and their view radii.
    def observe_arr(self, state, arr):
        x = state.p_pos[:, 0][:, None, None]
        y = state.p_pos[:, 1][:, None, None]
        half = self.view_rad_int
        offs = jnp.arange(-half, half + 1)
        dy   = offs[None, :, None]
        dx   = offs[None, None, :]
        xi   = jnp.clip(jnp.round(x + dx).astype(jnp.int16), 0, arr.shape[-1] - 1)
        yi   = jnp.clip(jnp.round(y + dy).astype(jnp.int16), 0, arr.shape[-1] - 1)
        return arr[:, yi, xi]

    # Accept coords and return distance to terrain points within a given chunk.
    # Output array is a square with odd dimensions, centred on the observer.
    def observe_hmap(self, state):
        z    = state.p_pos[:, 2][:, None, None]
        hgt  = self.observe_arr(state, state.hmap)
        dz   = hgt[0] * self.env_size - z
        dist = jnp.clip(dz / self.view_rad, -1.0, 1.0)
        dist = (dist + 1) / 2
        dist = jnp.where(hgt[1] == 1.0, dist, -dist)
        return dist
    
    # Quantise agent positions to grid coordinates.
    def quant_xy(self, agent_pos, grid_size):
        b  = grid_size
        s  = int(self.env_size / b)
        xi = jnp.floor(agent_pos[:, 0] / b).astype(jnp.int16)
        yi = jnp.floor(agent_pos[:, 1] / b).astype(jnp.int16)
        return jnp.clip(xi, 0, s - 1), jnp.clip(yi, 0, s - 1)

    # Convert agent pos to aerial image consisting of two layers (one per team).
    # Pixels represent map areas at the granularity specified by expl_grid_sz.
    # Pixel brightness is proportional to how close an agent is to the ground.
    # Pixels are zero where no agents are above that area.
    # If ONE_SWARM, the second layer is empty.
    def img_agents(self, agent_pos, hmap, dones):
        def img_team(agent_pos, hmap, dones, idxs):
            b  = self.expl_grid_sz
            s  = int(self.env_size / b)
            hm = jax.image.resize(hmap, (s, s), "bilinear").astype(PRE)
            ap = (agent_pos * ~dones[..., None])[idxs]
            xi, yi = self.quant_xy(ap, b)
            zi = jnp.clip(ap[:, 2] / s, 0.0, 1.0)
            im = jnp.zeros((s * s)).at[yi * s + xi].add(zi).reshape((s, s))
            return jnp.clip(
                jnp.where(im > hm, 1 - (im - hm), 0.0), min=0.0, max=1.0)
        # Stack team layers and return.
        team_0 = img_team(agent_pos, hmap, dones, self.team_0_idxs)
        team_1 = jnp.zeros_like(team_0) if ONE_SWARM else \
                 img_team(agent_pos, hmap, dones, self.team_1_idxs)
        return jnp.stack([team_0, team_1])

    # Determines which agents are oriented towards / facing others.
    def facing_other(self, diff_pos, ori):
        ohat = ori / (jnp.linalg.norm(ori, axis=-1, keepdims=True) + self.eps)
        d = jnp.linalg.norm(diff_pos, axis=-1) + self.eps
        u = diff_pos / d[..., None]
        return (jnp.einsum('ik,ijk->ij', ohat, u) >= self.fire_cos) # i facing j, over threshold

    # Determines which agents are facing the rears of others,
    # and therefore have an ideal shot.
    def facing_rear(self, diff_pos, ori):
        ohat = ori / (jnp.linalg.norm(ori, axis=-1, keepdims=True) + self.eps)
        d = jnp.linalg.norm(diff_pos, axis=-1) + self.eps
        u = diff_pos / d[..., None]
        a = jnp.einsum('ik,ijk->ij', ohat, u)  # i facing j
        b = jnp.einsum('jk,ijk->ij', ohat, u)  # j facing away from i
        return (a >= self.fire_cos) & (b >= self.fire_cos)

    def limit_speed(self, vel, max_speed):
        speed = jnp.linalg.norm(vel, axis=-1)
        scale = jnp.minimum(1.0, max_speed / (speed + self.eps))
        vel = vel * scale[..., None]
        return vel, jnp.linalg.norm(vel, axis=-1)

    # Hard-coded clay pigeons flee away from nearby agents while steering clear
    # of terrain and environment boundaries.
    def step_pigeons(self, state: State, pg_diff_pos, pg_sqr_dist, pg_land_heights):
        if self.num_pgon == 0:
            return state.pg_pos, state.pg_vel

        # Flee away from up to the 3 nearest visible agents for each pigeon.
        # pg_diff_pos is [A, P, 3], with invisible / invalid interactions zero-masked.
        max_threats = 3
        pg_diff = jnp.swapaxes(pg_diff_pos, 0, 1)      # [P, A, 3]
        pg_sqr = jnp.swapaxes(pg_sqr_dist, 0, 1)       # [P, A]
        pad_agents = max(0, max_threats - self.num_agnt)
        pg_sqr_pad = jnp.pad(pg_sqr, ((0, 0), (0, pad_agents)), constant_values=jnp.inf)
        pg_diff_pad = jnp.pad(pg_diff, ((0, 0), (0, pad_agents), (0, 0)))
        _, near_idxs = jax.lax.top_k(-pg_sqr_pad, max_threats)
        near_sqr = jnp.take_along_axis(pg_sqr_pad, near_idxs, axis=-1)
        near_diff = jnp.take_along_axis(pg_diff_pad, near_idxs[..., None], axis=1)
        near_valid = jnp.isfinite(near_sqr) & (near_sqr > 0.0)

        d = jnp.sqrt(jnp.maximum(near_sqr, 0.0)) + self.eps
        away = near_diff / d[..., None]
        away_w = 1.0 / d[..., None]
        flee = jnp.sum(jnp.where(near_valid[..., None], away * away_w, 0.0), axis=1)

        # Push up when too close to terrain. Keep a little lift to counter gravity.
        alt = state.pg_pos[:, -1] / self.env_size - pg_land_heights
        terr_push = jnp.clip((self.pgon_safe_alt / self.env_size - alt)
                             / (self.pgon_safe_alt / self.env_size + self.eps),
                             min=0.0, max=1.0)

        # Push inward when too close to the world boundary.
        edge_margin = self.view_rad
        low_edge  = jnp.clip((edge_margin - state.pg_pos) / (edge_margin + self.eps), 0.0, 1.0)
        high_xy   = jnp.clip((state.pg_pos[:, :2] - (self.env_size - edge_margin))
                             / (edge_margin + self.eps), 0.0, 1.0)
        high_z    = jnp.clip((state.pg_pos[:, 2] - (self.env_size * self.height_lim - edge_margin))
                             / (edge_margin + self.eps), 0.0, 1.0)
        edge_push = jnp.stack([
            low_edge[:, 0] - high_xy[:, 0],
            low_edge[:, 1] - high_xy[:, 1],
            low_edge[:, 2] - high_z,
        ], axis=-1)

        p_force = self.pgon_accel * flee
        p_force = p_force + edge_push * self.pgon_edge_gain
        p_force = p_force.at[:, 2].add(self.gravity + terr_push * self.pgon_terr_gain)
        p_force = jnp.where(state.pg_done[:, None], 0.0, p_force)

        pg_pos, pg_vel = self._integrate_state(p_force, state.pg_pos, state.pg_vel, state.pg_done)
        pg_vel, _ = self.limit_speed(pg_vel, self.pgon_max_speed)

        # Project any terrain penetration back up to the local surface.
        pg_norm_xy = pg_pos[:, :2] / self.env_size
        pg_land_new = sample_hmap(state.hmap, pg_norm_xy)
        pg_min_z = (pg_land_new + self.nrm_body_rad) * self.env_size
        penetrated = pg_pos[:, 2] < pg_min_z
        pg_pos = pg_pos.at[:, 2].set(jnp.maximum(pg_pos[:, 2], pg_min_z))
        pg_vel = pg_vel.at[:, 2].set(jnp.where(penetrated, jnp.maximum(pg_vel[:, 2], 0.0), pg_vel[:, 2]))

        return pg_pos.astype(PRE), pg_vel.astype(PRE)

    # Update agent attitudes given attitude update actions.
    def update_att(self, op_att, old_att, att_action):
        # If attitudes aren't in use, op_att still stores operator attitude, but all
        # drone attitudes are permanently set to initial/default attitudes.
        if not ATTITUDES:
            return self.init_att
        # If perfect attitude information, all atts permanently correct.
        if PRFCT_ATT:
            return jnp.concatenate(
                (jnp.full((self.ts, self.dim_a), op_att[0], dtype=PRE),
                 jnp.full((self.ts, self.dim_a), op_att[1], dtype=PRE))
            )
        # Update attitudes given actions.
        att = jnp.clip(old_att + att_action, min=0.0, max=1.0)
        # If one swarm, Agent 0's attitude == op_att[0].
        if ONE_SWARM:
            return jnp.concatenate((op_att[0, None, :], att[1:]))
        # Else (if two teams), assign separate operator intents to both team leads.
        return jnp.concatenate((op_att[0, None, :], att[1 : self.ts], 
                                op_att[1, None, :], att[self.ts + 1 :]))

    # STEP AND RESET ------------------------------------------------------- #

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: State, actions: dict):
        # Compute interactions.
        (
            diff_pos,
            sqr_dist,
            pg_diff_pos,
            pg_sqr_dist,
            agnt_colls,
            land_heights,
            land_colls,
            pg_land_heights,
            aerials,
            snapshots,
        ) = self.interactions(state)

        # Update exploration map with live agent positions.
        current_expl_map = self.img_agents(state.p_pos, state.hmap[0], state.done)
        expl_map = jnp.clip(current_expl_map + state.expl_map * self.expl_decay, max=1.0)
        
        # Move agents given actions. Set_actions zeroes out done agents.
        xyz, att, raw_actions = self.set_actions(actions, state.done)
        key, key_w = jax.random.split(key)
        p_pos, p_vel = self._world_step(key_w, state, xyz)

        # Move clay pigeons according to their hard-coded flee policy.
        pg_pos, pg_vel = self.step_pigeons(state, pg_diff_pos, pg_sqr_dist, pg_land_heights)
        
        # Stepwise update attitudes, not including the team leader/s,
        # which should always represent their pre-selected intent/s.
        # If agents can't act to update their preset attitudes,
        # return attitudes unchanged.
        attitudes = self.update_att(state.op_att, state.attitudes, att) \
                    if ATT_ACTNS else state.attitudes

        # Zero velocity for grounded agents.
        p_vel = p_vel * (~land_colls)[:, None].astype(PRE)

        # Limit velocity by max_speed.
        p_vel, scaled_speed = self.limit_speed(p_vel, self.max_speed)

        # Add new positions and velocities to history queues.
        pos_hist = jnp.concatenate(
            [jnp.expand_dims(p_pos, 1), state.pos_hist], axis=1
            )[:, :self.hist_q_len]
        vel_hist = jnp.concatenate(
            [jnp.expand_dims(p_vel, 1), state.vel_hist], axis=1
            )[:, :self.ori_hist]

        # Use velocity history to determine current orientation.
        # Don't update orientation for grounded drones.
        p_ori_update = jnp.mean(vel_hist, axis=1)
        p_ori = jnp.where(~land_colls[:, None], p_ori_update, state.p_ori)

        # Find given and received hits. Friendly fire is disallowed.
        # Only team 0 can fire in HIDE_SEEK mode.
        # Reset and increment fire_ago.
        correct_team  = (self.team_ids==0) if HIDE_SEEK \
                            else jnp.full((self.num_agnt,), True)
        within_fire   = (sqr_dist > 0.0) & (sqr_dist < self.fire_rad_sqr)
        facing_other  = self.facing_other(diff_pos, p_ori)
        fire_ready    = state.fire_ago == self.fire_refr
        zeros, ones   = jnp.zeros((self.ts, self.ts)), jnp.ones((self.ts, self.ts))
        friendly      = jnp.block([[ones, zeros], [zeros, ones]]).astype(bool)
        valid_fire    = (correct_team[:, None]
                        & within_fire
                        & facing_other
                        & fire_ready[:, None]
                        & ~ONE_SWARM 
                        & ~friendly)
        g_hits        = valid_fire.astype(PRE)
        r_hits        = g_hits.T
        total_agnt_g_hits = jnp.sum(g_hits, axis=-1)
        total_r_hits      = jnp.sum(r_hits, axis=-1)

        # Register agent -> clay pigeon firing.
        pg_within_fire = (pg_sqr_dist > 0.0) & (pg_sqr_dist < self.fire_rad_sqr)
        pg_facing      = self.facing_other(pg_diff_pos, p_ori)
        pg_valid_fire  = (correct_team[:, None]
                        & pg_within_fire
                        & pg_facing
                        & fire_ready[:, None])
        pg_hits        = pg_valid_fire.astype(PRE)
        total_pg_hits_by_agnt = jnp.sum(pg_hits, axis=-1)
        total_pg_hits_by_pgon = jnp.sum(pg_hits, axis=0)

        total_g_hits  = total_agnt_g_hits + total_pg_hits_by_agnt
        net_hits      = total_g_hits - total_r_hits
        fire          = total_g_hits > 0.0
        fire_ago      = jnp.clip(jnp.where(fire, 0, state.fire_ago + 1), max=self.fire_refr)

        # Update battery power given action costs and received hits.
        throttle_cost = jnp.sum(jnp.abs(xyz), axis=-1) * self.throt_batt
        fire_cost     = fire.astype(PRE) * self.fire_batt
        hit_cost      = total_r_hits * self.hit_batt
        batt_cost     = hit_cost #throttle_cost + fire_cost + hit_cost
        batt          = jnp.clip(state.batt - batt_cost, min=0.0)

        # Clay pigeons are two-hit kills by default.
        pg_batt = jnp.clip(state.pg_batt - total_pg_hits_by_pgon, min=0.0)
        pg_done_prev = state.pg_done

        # Post-move pigeon death bookkeeping only needs agent collisions and
        # boundary checks. Terrain penetration has already been projected away.
        pg_agnt_colls = self.pigeon_agent_collisions(p_pos, state.done, pg_pos, state.pg_done)
        pg_bound_colls = (jnp.any(pg_pos < 0.0, axis=-1) |
                          jnp.any(pg_pos[:, :-1] > self.env_size, axis=-1) |
                          (pg_pos[:, -1] > self.env_size * self.height_lim))
        pg_done = (pg_agnt_colls > 0) | pg_bound_colls | (pg_batt == 0.0) | state.pg_done

        # Update state. Only non-grounded agents move.
        state = state.replace(
            p_pos=p_pos.astype(PRE),
            p_vel=p_vel.astype(PRE),
            p_ori=p_ori.astype(PRE),
            pg_pos=pg_pos.astype(PRE),
            pg_vel=pg_vel.astype(PRE),
            pg_batt=pg_batt.astype(PRE),
            pg_done=pg_done.astype(bool),
            pos_hist=pos_hist.astype(PRE),
            vel_hist=vel_hist.astype(PRE),
            attitudes=attitudes.astype(PRE),
            actions=raw_actions.astype(PRE),
            last_expl_map=state.expl_map.astype(PRE),
            expl_map=expl_map.astype(PRE),
            snapshots=snapshots.astype(PRE),
            speed=scaled_speed.astype(PRE),
            batt=batt.astype(PRE),
            fire=fire.astype(bool),
            fire_ago=fire_ago.astype(int),
            step=(state.step + 1).astype(int),
            tot_steps=(state.tot_steps + 1).astype(int),
            r_hits=total_r_hits.astype(int),
            g_hits=total_g_hits.astype(int),
        )

        # Find environment boundary collisions.
        p = state.p_pos
        bound_colls = (jnp.any(p < 0.0, axis=-1) |                   # xyz under zero
                       jnp.any(p[:, :-1] > self.env_size, axis=-1) | # xy over env_size
                       (p[:, -1] > self.env_size * self.height_lim)) # z over height
        
        # Set dones according to collisions, battery levels, and timesteps.
        # Increment done_ago (counts the steps since an agent deactivated).
        agnt_done = (agnt_colls > 0) | land_colls | bound_colls | (batt == 0.0)
        epis_done = jnp.full((self.num_agnt), state.step >= self.max_steps)
        done  = agnt_done | epis_done | (state.done_ago > -1) | state.done
        dones = {a: done[i] for i, a in enumerate(self.agents)}
        dones.update({"__all__": jnp.all(done)})
        done_ago = jnp.where(state.done_ago > -1, state.done_ago + 1, -1)
        done_ago = jnp.clip(done_ago, max=self.death_anim)
        done_ago = jnp.where(done & (done_ago == -1), 0, done_ago)
        state = state.replace(done=done)
        state = state.replace(done_ago=done_ago)

        # Tally kills per agent, including clay pigeons.
        g_kills = jnp.sum(g_hits * (done_ago == 0)[None, :].astype(int), axis=-1)
        pg_kills = jnp.sum(pg_hits * ((pg_batt == 0.0) & ~pg_done_prev)[None, :].astype(int), axis=-1)
        g_kills = g_kills + pg_kills

        # Compute obs and rewards, and return.
        obs     = self.get_obs(diff_pos, sqr_dist, pg_diff_pos, pg_sqr_dist, aerials, state)
        rewards = self.get_rewards(sqr_dist, aerials, net_hits, g_kills, agnt_done, done_ago, state, train=not self.eval_mode)
        state   = state.replace(g_kills=g_kills.astype(int))
        return obs, state, rewards, dones, {}

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey, tot_steps: int) -> Tuple[chex.Array, State]:
        key_land, key_pos, key_att, key_pg = jax.random.split(key, 4)

        # Reset terrain heightmap to a different location.
        hmap = self.index_hmap(key_land)

        # Set agent positions. Ensure agents spawn above ground.
        norm_agnt_xy = jax.random.uniform(key_pos, (self.num_agnt, self.dim_p-1), dtype=PRE)
        norm_agnt_z  = jax.random.uniform(
            key_pos, self.num_agnt, dtype=PRE,
            minval=sample_hmap(hmap, norm_agnt_xy) + self.nrm_body_rad,
            maxval=self.height_lim - self.nrm_body_rad)
        agnt_pos = (jnp.concatenate([norm_agnt_xy, norm_agnt_z[:, None]], axis=-1)
                    * self.env_size)
        agnt_vel = jnp.zeros((self.num_agnt, self.dim_p), dtype=PRE)

        # Set clay pigeon positions. Ensure they spawn above ground.
        norm_pg_xy = jax.random.uniform(key_pg, (self.num_pgon, self.dim_p-1), dtype=PRE)
        norm_pg_z  = jax.random.uniform(
            key_pg, self.num_pgon, dtype=PRE,
            minval=sample_hmap(hmap, norm_pg_xy) + self.nrm_body_rad,
            maxval=self.height_lim - self.nrm_body_rad)
        pg_pos = (jnp.concatenate([norm_pg_xy, norm_pg_z[:, None]], axis=-1)
                  * self.env_size)
        pg_vel = jnp.zeros((self.num_pgon, self.dim_p), dtype=PRE)

        # Initialise pos and vel histories.
        hist     = lambda a, t : jnp.repeat(a[:, None, :], t, axis=1)
        pos_hist = hist(agnt_pos, self.hist_q_len)
        vel_hist = hist(agnt_vel, self.ori_hist)

        # Randomise operator intent/s. Sample from Dirichlet.
        shape  = (1,) if ONE_SWARM else (2,)
        alpha  = jnp.full(self.dim_a, self.dirich_alpha)
        op_att = jax.random.dirichlet(key_att, alpha, shape)

        # Reset agent attitudes. Provide the team leader/s knowledge of operator intent/s.
        attitudes = self.update_att(op_att, self.init_att, jnp.zeros_like(self.init_att))
        
        # Update warmup progress and multiplier. If self.warmup_on == False, warmup_multi == 0.
        warmup = jnp.clip((tot_steps / (self.warmup_steps + 1)).astype(PRE), max=1.0) # progress in [0, 1]
        warmup_multi = (1.0 - warmup) * self.warmup_on # multiplier in [0, 1]

        # Exploration map size.
        es = int(self.env_size / self.expl_grid_sz)

        # Return observations and env state.
        state = State(
            p_pos=agnt_pos.astype(PRE),
            p_vel=agnt_vel.astype(PRE),
            pg_pos=pg_pos.astype(PRE),
            pg_vel=pg_vel.astype(PRE),
            pg_batt=jnp.full((self.num_pgon,), self.pgon_init_batt, dtype=PRE),
            pg_done=jnp.full((self.num_pgon,), False).astype(bool),
            pos_hist=pos_hist.astype(PRE),
            vel_hist=vel_hist.astype(PRE),
            p_ori=jnp.zeros((self.num_agnt, self.dim_p), dtype=PRE),
            attitudes=attitudes.astype(PRE),
            actions=jnp.zeros((self.num_agnt, self.dim_p + self.dim_a), dtype=PRE),
            op_att=op_att.astype(PRE),
            speed=jnp.zeros((self.num_agnt), dtype=PRE),
            batt=jnp.full((self.num_agnt), self.init_battery, dtype=PRE),
            fire=jnp.full((self.num_agnt), False).astype(bool),
            fire_ago=jnp.full((self.num_agnt), self.fire_refr, dtype=int),
            hmap=hmap.astype(PRE),
            last_expl_map=jnp.zeros([2, es, es], dtype=PRE),
            expl_map=jnp.zeros([2, es, es], dtype=PRE),
            snapshots=jnp.zeros((self.num_agnt, self.view_width, self.view_width), dtype=PRE),
            step=0,
            tot_steps=tot_steps.astype(int),
            warmup=warmup.astype(PRE),
            warmup_multi=warmup_multi.astype(PRE),
            transition=False,
            done=jnp.full((self.num_agnt), False).astype(bool),
            done_ago=jnp.full((self.num_agnt), -1).astype(int),
            r_hits=jnp.zeros((self.num_agnt)).astype(int),
            g_hits=jnp.zeros((self.num_agnt)).astype(int),
            g_kills=jnp.zeros((self.num_agnt)).astype(int),
        )

        (diff_pos, sqr_dist, pg_diff_pos, pg_sqr_dist, _, _, _, _, aerials, snapshots) = self.interactions(state)
        state = state.replace(snapshots=snapshots)
        obs = self.get_obs(diff_pos, sqr_dist, pg_diff_pos, pg_sqr_dist, aerials, state)
        return obs, state

    # OBSERVATIONS AND REWARDS --------------------------------------------- #

    def get_obs(self, diff_pos, sqr_dist, pg_diff_pos, pg_sqr_dist, aerials, state: State
        ) -> Dict[str, chex.Array]:       
        # Agent observations:
        #   + self pos, vel, fire, team ID, battery
        #   + diff pos, vel, fire, and team IDs of closest neighbours
        #   + diff pos, vel, and battery of closest clay pigeons
        #   + aerial snapshot of terrain

        # Calculate ray angles, then rotate by agent yaw.
        W  = self.view_width
        cx = (PRE(W) - PRE(1.0)) * PRE(0.5)
        base_theta = PRE(2.0) * jnp.pi * (jnp.arange(W, dtype=PRE) / PRE(W))  # [R]
        yaw = jnp.arctan2(state.p_ori[:, 1], state.p_ori[:, 0]).astype(PRE)   # [N]
        theta = (yaw[:, None] + base_theta[None, :]).astype(PRE)[:, :, None]  # [N,R,1]

        # Samples along each ray from edge->center->edge (t in [-1, 1]).
        t = jnp.linspace(PRE(-1.0), PRE(1.0), W, dtype=PRE)[None, None, :]    # [1,1,S]

        # Ray coordinates in patch pixel space: (x,y) in [0, W-1]
        x = cx + (t * jnp.cos(theta) * cx)   # [N,R,S]
        y = cx + (t * jnp.sin(theta) * cx)   # [N,R,S]

        def _bilinear_sample(img: chex.Array, x: chex.Array, y: chex.Array) -> chex.Array:
            """img: [W,W], x/y: [R,S] float pixel coords -> out: [R,S]"""
            img = img.astype(PRE)
            x = jnp.clip(x, PRE(0.0), PRE(W - 1) - self.eps)
            y = jnp.clip(y, PRE(0.0), PRE(W - 1) - self.eps)

            x0 = jnp.floor(x).astype(jnp.int32)
            y0 = jnp.floor(y).astype(jnp.int32)
            x1 = jnp.minimum(x0 + 1, W - 1)
            y1 = jnp.minimum(y0 + 1, W - 1)

            wx = (x - x0.astype(PRE))
            wy = (y - y0.astype(PRE))

            Ia = img[y0, x0]
            Ib = img[y0, x1]
            Ic = img[y1, x0]
            Id = img[y1, x1]

            wa = (PRE(1.0) - wx) * (PRE(1.0) - wy)
            wb = wx * (PRE(1.0) - wy)
            wc = (PRE(1.0) - wx) * wy
            wd = wx * wy

            return wa * Ia + wb * Ib + wc * Ic + wd * Id

        # Vectorise bilinear sampling across agents: -> star: [N,R,S]
        star = jax.vmap(_bilinear_sample, in_axes=(0, 0, 0))(state.snapshots, x, y).astype(PRE)
        star_flat = star.reshape((self.num_agnt, W * W))  # [N, R*S]

        # Find closest-neighbour indices for all agents. Always reserve
        # self.obs_slots slots and pad with invalid entries when there are
        # fewer real neighbours than slots.
        pad_nbor = max(0, self.obs_slots - self.num_agnt)
        dists = jnp.where(sqr_dist > 0.0, sqr_dist, jnp.inf)
        dists_pad = jnp.pad(dists, ((0, 0), (0, pad_nbor)), constant_values=jnp.inf)
        diff_pos_pad = jnp.pad(diff_pos, ((0, 0), (0, pad_nbor), (0, 0)))
        p_vel_pad = jnp.pad(state.p_vel, ((0, pad_nbor), (0, 0)))
        attitudes_pad = jnp.pad(state.attitudes, ((0, pad_nbor), (0, 0)))
        team_ids_pad = jnp.pad(self.team_ids, (0, pad_nbor))
        fire_pad = jnp.pad(state.fire.astype(PRE), (0, pad_nbor))
        _, nbor_idxs_all = jax.lax.top_k(-dists_pad, self.obs_slots)
        dists_k_all = jnp.take_along_axis(dists_pad, nbor_idxs_all, -1)

        # Find closest clay-pigeon indices for all agents. Again, always
        # reserve self.obs_slots slots and zero-pad when pigeons are absent
        # or fewer than the preset slot count.
        pad_pgon = max(0, self.obs_slots - self.num_pgon)
        pg_dists = jnp.where(pg_sqr_dist > 0.0, pg_sqr_dist, jnp.inf)
        pg_dists_pad = jnp.pad(pg_dists, ((0, 0), (0, pad_pgon)), constant_values=jnp.inf)
        pg_diff_pos_pad = jnp.pad(pg_diff_pos, ((0, 0), (0, pad_pgon), (0, 0)))
        pg_vel_pad = jnp.pad(state.pg_vel, ((0, pad_pgon), (0, 0)))
        pg_batt_pad = jnp.pad(state.pg_batt, (0, pad_pgon))
        _, pgon_idxs_all = jax.lax.top_k(-pg_dists_pad, self.obs_slots)
        pg_dists_k_all = jnp.take_along_axis(pg_dists_pad, pgon_idxs_all, -1)

        def _obs(i: int):
            # Indices of closest neighbours, nearest first.
            nbor_idxs = nbor_idxs_all[i]

            # Self observations, normalised to [-1, 1].
            self_pos = (state.p_pos[i] / self.env_size) * 2 - 1
            self_vel =  state.p_vel[i] / self.max_speed
            self_att =  state.attitudes[i] * 2 - 1
            self_fre =  state.fire[i, None].astype(PRE) * 2 - 1
            self_frd = (state.fire_ago[i, None] / self.fire_refr).astype(PRE) * 2 - 1
            self_tem =  self.team_ids[i, None] * 2 - 1
            self_bat = (state.batt[i, None] / self.init_battery) * 2 - 1
            self_act =  state.actions[i] # Last action taken

            # Delta to closest neighbours, normalised to [-1, 1].
            # Hide opponents' attitudes.
            nbor_pos = diff_pos_pad[i, nbor_idxs] / self.view_rad
            nbor_vel = (p_vel_pad[nbor_idxs] - state.p_vel[i]) / self.max_speed
            diff_tem = jnp.abs(team_ids_pad[nbor_idxs] - self.team_ids[i]).astype(PRE)
            nbor_att = (attitudes_pad[nbor_idxs] - state.attitudes[i]) * (PRE(1.0) - diff_tem)[:, None]
            nbor_fre = fire_pad[nbor_idxs] * 2 - 1
            nbor_tem = diff_tem

            # Note any out-of-range neighbour idxs.
            oor_idxs = dists_k_all[i] == jnp.inf

            # Mask out-of-range neighbour info.
            zero_2d  = lambda a : jnp.where(oor_idxs[:, None], 0.0, a)
            zero_1d  = lambda a : jnp.where(oor_idxs         , 0.0, a)
            nbor_pos = zero_2d(nbor_pos)
            nbor_vel = zero_2d(nbor_vel)
            nbor_att = zero_2d(nbor_att)
            nbor_fre = zero_1d(nbor_fre)
            nbor_tem = zero_1d(nbor_tem)

            # Rotation-invariant nbor features expressed in the same frame as nbor_pos/nbor_vel.
            # Norms scaled by sqrt(3) so that max norm of a [-1,1]^3 vector maps to ~1.
            sqrt3 = jnp.sqrt(PRE(3.0))
            pos_norm = jnp.linalg.norm(nbor_pos, axis=-1)
            vel_norm = jnp.linalg.norm(nbor_vel, axis=-1)
            # Range, speed.
            nbor_rng = (jnp.clip(pos_norm / (sqrt3 + self.eps), 0.0, 1.0) * PRE(2.0) - PRE(1.0))
            nbor_spd = (jnp.clip(vel_norm / (sqrt3 + self.eps), 0.0, 1.0) * PRE(2.0) - PRE(1.0))
            # Radial speed.
            r_hat = nbor_pos / (pos_norm[:, None] + self.eps)
            radial = jnp.sum(r_hat * nbor_vel, axis=-1)
            nbor_rdot = jnp.clip(radial / (sqrt3 + self.eps), -1.0, 1.0)
            # Tangential speed.
            u_perp = nbor_vel - radial[:, None] * r_hat
            tang = jnp.linalg.norm(u_perp, axis=-1)
            nbor_tan = (jnp.clip(tang / (sqrt3 + self.eps), 0.0, 1.0) * PRE(2.0) - PRE(1.0))

            # Zero invariants for out-of-range neighbours (padding slots). Stack.
            nbor_rng  = zero_1d(nbor_rng)
            nbor_spd  = zero_1d(nbor_spd)
            nbor_rdot = zero_1d(nbor_rdot)
            nbor_tan  = zero_1d(nbor_tan)
            nbor_inv  = jnp.stack([nbor_rng, nbor_spd, nbor_rdot, nbor_tan], axis=-1)

            # Pairwise formation geometry: nbor-nbor distances.
            # Compute on normalised positions, then scale by 2*sqrt(3) so max separation -> ~1.
            present = ~oor_idxs
            dmat = jnp.linalg.norm(nbor_pos[:, None, :] - nbor_pos[None, :, :], axis=-1)
            pair_d = dmat[self.pair_i, self.pair_j]  # (P,)
            pair_m = present[self.pair_i] & present[self.pair_j]
            pair_feat = (jnp.clip(pair_d / (PRE(2.0) * sqrt3 + self.eps), 0.0, 1.0) * PRE(2.0) - PRE(1.0))
            pair_feat = jnp.where(pair_m, pair_feat, PRE(0.0))

            # Concatenate self attributes.
            self_ = jnp.concatenate([
                self_pos, # position
                self_vel, # velocity
                self_att, # attitude
                self_fre, # just fired
                self_frd, # fire recharge progress
                self_tem, # team
                self_bat, # battery level
                self_act, # last action
                ])

            # Concatenate neighbour attributes and flatten.
            nbor_ = jnp.concatenate([
                nbor_pos,
                nbor_vel, 
                nbor_att, 
                nbor_fre[:, None], 
                nbor_tem[:, None], 
                nbor_inv, # invariant features
                ], axis=-1).flatten()

            # Append pairwise geometry (10 scalars, if self.obs_slots == 5).
            nbor_ = jnp.concatenate([nbor_, pair_feat], axis=-1)

            # Clay pigeons: separate channel so agent-agent pair geometry remains clean.
            pgon_idxs = pgon_idxs_all[i]
            pgon_pos = pg_diff_pos_pad[i, pgon_idxs] / self.view_rad
            pgon_vel = (pg_vel_pad[pgon_idxs] - state.p_vel[i]) / self.pgon_max_speed
            pgon_bat = (pg_batt_pad[pgon_idxs] / self.pgon_init_batt) * 2 - 1

            pg_oor = pg_dists_k_all[i] == jnp.inf
            pg_zero_2d = lambda a : jnp.where(pg_oor[:, None], 0.0, a)
            pg_zero_1d = lambda a : jnp.where(pg_oor         , 0.0, a)
            pgon_pos = pg_zero_2d(pgon_pos)
            pgon_vel = pg_zero_2d(pgon_vel)
            pgon_bat = pg_zero_1d(pgon_bat)

            pg_pos_norm = jnp.linalg.norm(pgon_pos, axis=-1)
            pg_vel_norm = jnp.linalg.norm(pgon_vel, axis=-1)
            pg_r_hat = pgon_pos / (pg_pos_norm[:, None] + self.eps)
            pg_radial = jnp.sum(pg_r_hat * pgon_vel, axis=-1)
            pg_u_perp = pgon_vel - pg_radial[:, None] * pg_r_hat
            pg_tang = jnp.linalg.norm(pg_u_perp, axis=-1)

            pg_rng = (jnp.clip(pg_pos_norm / (sqrt3 + self.eps), 0.0, 1.0) * PRE(2.0) - PRE(1.0))
            pg_spd = (jnp.clip(pg_vel_norm / (sqrt3 + self.eps), 0.0, 1.0) * PRE(2.0) - PRE(1.0))
            pg_rdot = jnp.clip(pg_radial / (sqrt3 + self.eps), -1.0, 1.0)
            pg_tan = (jnp.clip(pg_tang / (sqrt3 + self.eps), 0.0, 1.0) * PRE(2.0) - PRE(1.0))

            pg_inv = jnp.stack([
                pg_zero_1d(pg_rng),
                pg_zero_1d(pg_spd),
                pg_zero_1d(pg_rdot),
                pg_zero_1d(pg_tan),
            ], axis=-1)

            pgon_ = jnp.concatenate([
                pgon_pos,
                pgon_vel,
                pgon_bat[:, None],
                pg_inv,
            ], axis=-1).flatten()

            # Flatten and scale aerials.
            # aeri_ = state.snapshots[i].flatten() * 2 - 1
            aeri_ = star_flat[i] * 2 - 1

            # Concatenate self, neighbour, and terrain obs.
            return jnp.concatenate([self_, nbor_, pgon_, aeri_])
        
        return {a: _obs(i) for i, a in enumerate(self.agents)}

    def get_rewards(
            self, sqr_dist, aerials, net_hits, kills, agnt_done, done_ago, state, train=True,
            ) -> Dict[str, chex.Array]:
        # Scaling preparation.
        # Define warm-up multiplier (in [0, 1]).
        # Used to scale reward terms as a function of training progress.
        w = state.warmup_multi
        # Define reward scaling for attitude-dependent terms, using
        # both warm-up and team leads' attitudes.
        scale_team_0 = lambda t : w + self.rew_scaling * state.op_att[0, t]
        scale_team_1 = lambda t : w + self.rew_scaling * state.op_att[1, t]
        coeffs = jnp.asarray(
            [list(map(scale_team_0, range(self.dim_a))),
             list(map(scale_team_1, range(self.dim_a)))]
        )

        # Cohesion term preparation.
        sqr_dist_norm = jnp.where(sqr_dist > 0.0, sqr_dist, self.view_rad_sqr) / self.view_rad_sqr
        sqr_dist_norm_safe = jnp.clip(sqr_dist_norm, min=self.safe_dist ** 2)

        # Attitude term preparation.
        cos_sim = lambda a, b : jnp.dot(a, b) / ((jnp.linalg.norm(a) * jnp.linalg.norm(b)) + self.eps)

        # Occupation term preparation.
        mid = aerials.shape[-1] // 2
        hgt = aerials[:, mid, mid]
        roi = hgt > 0.0
        
        # Scouting term preparation.
        xi,yi = self.quant_xy(state.p_pos, self.expl_grid_sz)
        map_coverage = state.expl_map.mean(axis=(1,2))

        # Per-agent reward function.
        # Rewards are largely local/individual and positive.
        def _rew(i: int):
            # Find which team this agent is on, as well as this agent's 
            # team lead's attitudes.
            team = self.team_ids[i]

            # AUXILIARY ---------------------------------------------------- #

            # Cohesion / distance to neighbours. If no neighbours in range, no reward.
            # Reward maxes-out at a safe distance to avoid clustering too tightly.
            # cohesion_term = (1 - jnp.mean(sqr_dist_norm_safe[i]))

            # Speed.
            # speed_term = state.speed[i] / self.max_speed

            # ATTITUDE ----------------------------------------------------- #

            # Attitude sync (operator / team leader).
            sync_term = cos_sim(state.attitudes[i], state.op_att[team]) ** 2
            
            # COMBAT ------------------------------------------------------- #
            # + net_hits (hits landed - hits received)
            # + kills    (enemies shot down)
            combat_term = net_hits[i] + kills[i]

            # SCOUTING ----------------------------------------------------- #
            # Local: how explored its current position is in comparison to t-1.
            locl_sc = (state.expl_map - state.last_expl_map)[team, yi[i], xi[i]]
            # Global: fraction of the map covered by this agent's team.
            # glbl_sc = map_coverage[team]
            # scouting_term = ((locl_sc + glbl_sc) / 2.0).reshape(())
            scouting_term = locl_sc.reshape(())

            # if an agent enters multiple fresh chunks, how might it be rewarded?
            # enumerate chunks crossed in last timestep?
            # also, aren't agents incentivised to travel diagonally to maximise Manhattan?
            # What we actually want: agents to spread out and map areas
            # - individual reward: 

            # OCCUPATION --------------------------------------------------- #
            # Is this agent directly overhead a ROI?
            # TODO: Add team_roi_coverage
            # occup = overhead_roi * dist_from_ground * team_roi_coverage
            occupation_term = roi[i] * (1 - jnp.abs(hgt[i])) #* roi_coverage[i]

            # CONSERVATION ------------------------------------------------- #

            # Battery cost.
            # conservation_term = state.batt[i] / self.init_battery

            # LAW ---------------------------------------------------------- #

            # Self destroyed. agnt_done is used instead of state.done, else,
            # agents would always receive negative reward at episode end.
            destroyed_term = agnt_done[i].astype(PRE)

            # -------------------------------------------------------------- #
            reward = (

            # Primary
                # Only reward attitude sync during training
                + self.oper_rew * sync_term * (train and ATTITUDES and ATT_ACTNS and not PRFCT_ATT)
                - self.dest_rew * destroyed_term
            # Auxiliary / warm-up
                # + self.cohe_rew * w * cohesion_term
                # + self.sped_rew * w * speed_term
                # + self.cons_rew * (1-w) * conservation_term
            # Attitude-scaled
                + self.cmbt_rew * coeffs[team, 0] * combat_term
                + self.occp_rew * coeffs[team, 1] * occupation_term
                + self.sctg_rew * coeffs[team, 2] * scouting_term

            ).astype(jnp.float32)

            # Ensure already done agents do not receive reward signal.
            reward = jnp.where(done_ago[i] > 0, 0.0, reward)

            # Ensure negative utility of an action cannot be worse than death.
            reward = jnp.clip(reward, min=-self.dest_rew)
            
            return reward

        return {a: _rew(i) for i, a in enumerate(self.agents)}

    # ACTIONS AND FORCES --------------------------------------------------- #

    # Process actions, zeroing where agents are done.
    def set_actions(self, actions: Dict, done: chex.Array):
        actions = jnp.array([actions[i] for i in self.agents]).reshape(
            (self.num_agnt, -1)) * ~done[:, None] * ~self.frozen[:, None]
        xyz = actions[:, :self.dim_p] * self.accel
        att = actions[:, self.dim_p:self.dim_p + self.dim_a] * self.att_update
        return xyz, att, actions

    # Update agent movement. Return p_pos and p_vel.
    def _world_step(self, key: chex.PRNGKey, state: State, u: chex.Array):
        key_noise = jax.random.split(key, self.num_agnt)
        # p_force = self._apply_action_noise(key_noise, u)
        # return self._integrate_state(p_force, state.p_pos, state.p_vel)
        return self._integrate_state(u, state.p_pos, state.p_vel, self.frozen)

    # Apply noise to actions (amount specified by u_noise).
    @partial(jax.vmap, in_axes=[None, 0, 0])
    def _apply_action_noise(self, key: chex.PRNGKey, u: chex.Array):
        return u + jax.random.normal(key, shape=u.shape) * self.u_noise

    # Integrate physical state.
    @partial(jax.vmap, in_axes=[None, 0, 0, 0, 0])
    def _integrate_state(self, p_force, p_pos, p_vel, frozen):
        p_pos += p_vel * self.dt
        p_vel  = p_vel * (1 - self.damping)
        p_vel += (p_force / self.mass) * self.dt
        p_vel -= jnp.array([0, 0, self.gravity]) * ~frozen[None]
        return p_pos, p_vel

    # RENDER - PYVISTA VERSION --------------------------------------------- #
    
    def _render_view(
        self,
        terrain: pv.StructuredGrid,
        pos,
        agnt_glyphs_0: pv.PolyData,
        agnt_glyphs_1: pv.PolyData,
        pgon_glyphs: Optional[pv.PolyData],
        view_glyphs: pv.PolyData,
        hist,
        attitude_colors,
        deat_fade,
        window_size: Tuple[int, int],
        view: str,
        camera_pos: Tuple[Tuple[float, float, float],
                          Tuple[float, float, float],
                          Tuple[float, float, float]],
        transition: bool,
        background: bool = True,
        clip_bounds: bool = True,
        toggle_bounds: bool = True,
        toggle_terrain: bool = True,
        toggle_agent_glyphs: bool = True,
        toggle_view_glyphs: bool = True,
        toggle_agent_lights: bool = True,
        toggle_trails: bool = True,
        toggle_labels: bool = True,
    ) -> np.ndarray:
        # Initialise plotter.
        pl = pv.Plotter(off_screen=True, window_size=window_size,
                        lighting='none')#'three lights' if view=='perspective' else 'none')
        pl.set_background('black', top='grey' if background else 'black')

        # Add terrain volume, if obstacles allowed. Else, render bounding box.
        if toggle_terrain and not KILL_BOX:
            if view == 'side':
                pl.add_mesh(terrain,
                            color='lightgrey',
                            opacity=0.1,
                            show_edges=True,
                            smooth_shading=False,
                            lighting=False,
                            style='wireframe',
                            line_width=0.5)
            else:
                pl.add_mesh(terrain,
                            scalars='roi',
                            cmap=[LAND_CLR, LDMK_CLR],
                            opacity=1.0,
                            show_edges=True,
                            smooth_shading=True)
        else:
            pl.add_mesh(pv.Box(self.bounds),
                        color='lightgrey',
                        opacity=0.1,
                        show_edges=True,
                        smooth_shading=False,
                        lighting=False,
                        style='wireframe',
                        line_width=0.5)

        # Set camera and lights.
        pl.camera_position = camera_pos
        if view == 'top':
            pl.enable_parallel_projection()
            pl.enable_lightkit()
            if toggle_bounds:
                pl.show_bounds(
                    show_zaxis=False, 
                    use_3d_text=False, 
                    bold=False,
                    font_size=8, 
                    xtitle='x', 
                    ytitle='y', 
                    color='white')
        elif view == 'side':
            pl.enable_parallel_projection()
            pl.enable_lightkit()
            if toggle_bounds:
                pl.show_bounds(
                    show_xaxis=False, 
                    use_3d_text=False, 
                    bold=False,
                    font_size=8, 
                    ytitle='y', 
                    ztitle='z', 
                    color='white')
        elif view == 'perspective':
            intensities = [1, 0.5]
            all_angles = [(45.0, 45.0), (-30.0, 60.0)]
            def _to_pos(elevation: float, azimuth: float) -> tuple[float, float, float]:
                theta = azimuth * np.pi / 180.0
                phi = (90.0 - elevation) * np.pi / 180.0
                x = np.sin(theta) * np.sin(phi)
                y = np.cos(phi)
                z = np.cos(theta) * np.sin(phi)
                return x, y, z
            for intensity, angles in zip(intensities, all_angles):
                light = pv.Light(light_type='camera light')
                light.intensity = intensity
                light.position = _to_pos(*angles)
                pl.add_light(light)

        # Add agent meshes and visibility billboards.
        if clip_bounds:
            clip_mesh     = lambda g : g.clip_box(self.bounds, invert=False).extract_surface()
            view_glyphs   = clip_mesh(view_glyphs)
            agnt_glyphs_0 = clip_mesh(agnt_glyphs_0)
            agnt_glyphs_1 = clip_mesh(agnt_glyphs_1)
            pgon_glyphs   = None if pgon_glyphs is None else clip_mesh(pgon_glyphs)
        if toggle_view_glyphs:
            pl.add_mesh(view_glyphs, scalars="RGB", rgb=True, smooth_shading=False, 
                style='wireframe', lighting=False, line_width=0.25, ambient=0.5)
        if toggle_agent_glyphs:
            shading = view == 'perspective'
            pl.add_mesh(agnt_glyphs_0, scalars="RGB", rgb=True, smooth_shading=False,
                show_edges=False, pbr=shading, ambient=0.9)
            pl.add_mesh(agnt_glyphs_1, scalars="RGB", rgb=True, smooth_shading=False,
                show_edges=False, pbr=shading, ambient=0.9)
            if pgon_glyphs is not None:
                pl.add_mesh(pgon_glyphs, scalars="RGB", rgb=True, smooth_shading=False,
                    show_edges=False, pbr=shading, ambient=0.9)

        # Add per-agent lighting, sky trails, and labels.
        omni_light = lambda p, c, i : pv.Light(
            position=p,
            positional=True,
            intensity=0.5 * i,
            cone_angle=360,
            shadow_attenuation=0.0,
            attenuation_values=(0.025, 0.0, 0.015),
            color=c)
        # Lighting.
        if toggle_agent_lights:
            for n,p in enumerate(pos):
                pl.add_light(omni_light(p, attitude_colors[n], float(deat_fade[n])))
        # Trails.
        if (not transition) and toggle_trails:
            for n,p in enumerate(pos):
                trail = np.compress(np.sum(hist[n], axis=1) != 0, hist[n], axis=0)
                if len(trail) > 1:
                    trail_lines = pv.lines_from_points(trail)
                    pl.add_mesh(trail_lines, color=attitude_colors[n], line_width=2)
        # Labels.
        if (not transition) and toggle_labels:
            labels = [str(i + 1) for i in range(pos.shape[0])]
            lifted_pos = np.concatenate([pos[:, :2], pos[:, 2:] + self.body_rad * 2], axis=-1)
            # Team 0.
            pl.add_point_labels(
                lifted_pos[:self.ts], 
                labels[:self.ts], 
                font_size=11, 
                text_color='white', 
                shape_color=AGNT_CLR, 
                show_points=False, 
                bold=True, 
                shadow=True, 
                tolerance=1.0
            )
            # Team 1.
            pl.add_point_labels(
                lifted_pos[self.ts:], 
                labels[self.ts:], 
                font_size=11, 
                text_color='white', 
                shape_color=AGNT_CLR if ONE_SWARM else ADVR_CLR, 
                show_points=False, 
                bold=True, 
                shadow=True, 
                tolerance=1.0
            )

        # Render and return.
        img = pl.screenshot(transparent_background=not background, return_img=True)
        pl.close()
        return img[..., :3].astype(np.uint8)
    
    def _render_info_panel(
        self,
        size: Tuple[int, int],
        state,
        agnt_colors,
        step_str,
    ) -> np.ndarray:
        img  = Image.new("RGB", size, (0, 0, 40))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("DejaVuSansMono.ttf", 12)
        txt_pos = (25, 32)
        spacing = 6
        line_height = 12
        div_str = f"\n{"_"*35}\n\n"
        
        # Write header.
        txt = f"\n\nAGENT STATUS:\n\n+-----REC-FIR-KLL-------BATTERY---+"
        draw.multiline_text(txt_pos, txt, font=font, spacing=spacing, fill='white')

        # Write info lines per agent.
        curr_line = 7
        for i in range(min(self.num_agnt, 10)):
            fire_recharge = (10 * state.fire_ago[i]) // self.fire_refr
            c   = tuple((agnt_colors[i] * 255).astype(int))
            re  = '◼' if state.r_hits[i]  > 0 else '☐'
            kl  = '◼' if state.g_kills[i] > 0 else '☐'
            fr  = '◼' if state.g_hits[i]  > 0 else '☐' \
                if fire_recharge == 10 else str(fire_recharge)
            ba  = ("OFFLINE---" if bool(state.done[i]) else 
                  ('▮' * round(10 * state.batt[i] / self.init_battery) + 
                   '▯' * (10 - round(10 * state.batt[i] / self.init_battery))))
            txt = f"A {i+1:>2}:  {re}   {fr}   {kl}       [{ba}]"
            draw.multiline_text((txt_pos[0], txt_pos[1] + line_height * curr_line),
                                 txt, font=font, spacing=spacing, fill=c, 
                                 stroke_width=float(state.done[i]), stroke_fill='red')
            curr_line += 1
        draw.multiline_text((txt_pos[0], txt_pos[1] + line_height * curr_line),
            f"+{'-'*33}+", font=font, spacing=spacing, fill='white')
        
        # Write info about exploration, episode, and timestep.
        perc = np.asarray(state.expl_map, dtype=np.float32).mean(axis=(1,2)) * 100
        exp_txt = int(perc[0]) if ONE_SWARM else f"{int(perc[0])}% | {int(perc[1])}%"
        txt = f"\nMAP COVERAGE: {exp_txt}{div_str}{step_str}{div_str}TERRAIN SENSORS:"
        draw.multiline_text((txt_pos[0], txt_pos[1] + line_height * curr_line),
            txt, font=font, spacing=spacing, fill='white')
        curr_line += 13
        
        # Display aerial observations made by agents as coloured heatmaps:
        # - regular terrain appears brown, while regions of interest are blue-green
        # - far-above terrain is black, far-below is white
        obs_scale = 0.411
        img_pos_list = []
        for i in range(2):
            for j in range(5):
                offset = int(size[0] * obs_scale) + 5
                img_pos_list.append((int(size[0] * 0.088) + i * offset,
                    txt_pos[1] + line_height * curr_line + j * offset))
        img_pos_list = img_pos_list[:min(self.num_agnt, 10)]
        for i,p in enumerate(img_pos_list):
            heat = plt.get_cmap('BrBG')(np.asarray(state.snapshots[i], dtype=np.float32))
            if float(state.done[i]):
                heat *= 0.5
                heat[..., 0] = np.eye(heat.shape[0])
            heat = Image.fromarray((heat * 255).astype(np.uint8))
            new_size = (int(size[0] * obs_scale), int(size[0] * obs_scale))
            resized  = heat.resize(new_size, Image.Resampling.NEAREST)
            img.paste(resized.transpose(Image.FLIP_TOP_BOTTOM), p)
            draw.multiline_text(p, f' A{i + 1}', font=font,
                spacing=spacing, fill='white', stroke_width=1, stroke_fill='black')
        return np.array(img, dtype=np.uint8)

    def _render_expl_map(
        self,
        size: Tuple[int, int],
        state,
    ) -> np.ndarray:
        img  = Image.new("RGB", size, 'black')
        draw = ImageDraw.Draw(img)
        pos = (60, 60)
        font = ImageFont.truetype("DejaVuSansMono.ttf", 12)
        obs_scale = 0.775
        # Display exploration heatmap.
        expl = np.asarray(state.expl_map, dtype=np.float32)
        heat = plt.get_cmap('cividis')(expl[0]) if ONE_SWARM else \
               plt.get_cmap('managua')((expl[0] - expl[1]) + 1 / 2)
        heat = Image.fromarray((heat * 255).astype(np.uint8))
        new_size = (int(size[0] * obs_scale), int(size[0] * obs_scale))
        resized  = heat.resize(new_size, Image.Resampling.NEAREST)
        img.paste(resized.transpose(Image.FLIP_TOP_BOTTOM), pos)
        return np.array(img, dtype=np.uint8)

    def build_glyphs(self, colors, pos, ori, scale, geom='disc'):
        # Initialise glyph points.
        pts = pv.PolyData(pos)
        if type(scale) == float:
            pts["glyph_scale"] = np.full((pos.shape[0],), scale)
        else:
            pts["glyph_scale"] = scale
        rgb = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
        pts["RGB"] = rgb

        # Select geometry.
        if geom=='cone':
            geom = pv.Cone(radius=1.0, height=2.5, resolution=6)
        elif geom=='disc':
            geom = pv.Disc(inner=1.0, outer=1.0, c_res=30, normal=(1.0, 0.0, 0.0))  
        elif geom=='sphere':
            geom = pv.Sphere(radius=0.8, phi_resolution=9, theta_resolution=9)
        pts["orient"] = ori
        orient = "orient"

        # Return glyphs.
        return pts.glyph(geom=geom, scale="glyph_scale", orient=orient, factor=1.0)

    def draw_title_and_border(self, img, title, thick=1, clr=(255,255,255)):
        im   = Image.fromarray(img.astype(np.uint8))
        w, h = im.size
        font = ImageFont.truetype("DejaVuSansMono.ttf", 15)
        draw = ImageDraw.Draw(im)
        draw.rectangle([(0, 0), (w, thick)], fill=clr)
        draw.rectangle([(0, h - thick), (w, h)], fill=clr)
        draw.rectangle([(0, thick), (thick, h - thick)], fill=clr)
        draw.rectangle([(w - thick, thick), (w, h - thick)], fill=clr)
        draw.multiline_text((25, 25), title, font=font, spacing=6, fill=clr)
        return np.array(im, dtype=np.uint8)

    def render_vtk(
        self,
        state,
        step_str,
        azim: float = 0.0,
        h: int = RENDER_H,
    ) -> np.ndarray:
        # Build terrain volume from heightmap.
        pv.global_theme.show_scalar_bar = False
        z     = np.asarray(state.hmap[0], dtype=np.float32) * self.env_size
        x, y  = np.meshgrid(np.linspace(0.0, self.env_size, z.shape[1]), 
                            np.linspace(0.0, self.env_size, z.shape[0]))
        grid  = pv.StructuredGrid(x, y, z)
        top   = grid.points.copy()
        bot   = grid.points.copy()
        bot[:, -1] = 0.0
        terrain = pv.StructuredGrid()
        terrain.points = np.vstack((bot, top))
        roi_top = state.hmap[1].astype(np.uint8).ravel(order="F")
        roi_bot = np.zeros_like(roi_top)
        terrain.point_data["roi"] = np.concatenate([roi_bot, roi_top], axis=-1)
        terrain.dimensions = [*grid.dimensions[0:2], 2]

        # Assign colours given agent states.
        # Progressively add/fade colours given fire and death sequences.
        colors = np.tile(clr.to_rgb(AGNT_CLR), (self.num_agnt, 1))
        colors[self.team_ids > 0] = clr.to_rgb(ADVR_CLR)
        fade = lambda f, c, c_arr : \
            (1 - f[:, None]) * np.asarray(clr.to_rgb(c))[None, :] + f[:, None] * c_arr
        fire_fade  = state.fire_ago / self.fire_refr
        deat_fade  = jnp.where(state.done_ago > -1, 1 - state.done_ago / self.death_anim, 1)
        colors     = fade(fire_fade, FIRE_CLR, colors)
        colors     = fade(deat_fade, DONE_CLR, colors)
        colors     = np.asarray(colors, dtype=np.float32)

        # If attitudes aren't directly mappable to RGB (i.e. != 3 objectives),
        # pad or truncate.
        if self.dim_a < 3:
            att_colors = np.pad(
                np.asarray(state.attitudes, dtype=np.float32),
                ((0, 0), (0, 3 - self.dim_a)),
            )
        else:
            att_colors = np.asarray(state.attitudes[:, :3], dtype=np.float32)
        att_colors = np.asarray(fade(fire_fade, FIRE_CLR, att_colors))
        att_colors = np.asarray(fade(deat_fade, DONE_CLR, att_colors))
        att_colors = att_colors if (CLRMP_ATT and ATTITUDES) else colors

        # Set cameras (perspective + top and side ortho).
        # p_center and p_dist allow for the perspective camera to move and zoom.
        # Both are determined by a rolling peak-to-peak average, measuring swarm spread 
        # in xy across roll_steps to avoid jittery motion.
        pos        = np.asarray(state.p_pos,    dtype=np.float32)
        ori        = np.asarray(state.p_ori,    dtype=np.float32)
        hist       = np.asarray(state.pos_hist, dtype=np.float32)
        pg_pos     = np.asarray(state.pg_pos,   dtype=np.float32)
        pg_vel     = np.asarray(state.pg_vel,   dtype=np.float32)
        pg_done    = np.asarray(state.pg_done,  dtype=bool)
        roll_steps = 25
        min_dist   = self.env_size * 2
        pos_window = hist[:, :roll_steps]
        roll_ptp   = np.mean(np.ptp(pos_window[..., :2], axis=0))
        p_center   = np.mean(pos_window, axis=(0,1))
        p_dist     = max(roll_ptp * 3.0, min_dist)
        center     = np.array([self.env_size / 2.0, self.env_size / 2.0, self.env_size * self.height_lim / 2.0])
        s_center   = center if KILL_BOX else \
                     np.array([self.env_size / 2.0, self.env_size / 2.0, self.env_size * self.height_lim / 1.5])
        dist       = self.env_size * 2.5
        yaw        = np.deg2rad(azim)
        pitch      = np.deg2rad(55)
        cam        = np.array([np.cos(yaw) * np.sin(pitch),
                               np.sin(yaw) * np.sin(pitch),
                               np.cos(pitch)])
        per_pos = tuple((p_center + cam * p_dist).tolist())
        top_pos = (center[0], center[1], center[2] + dist)
        sid_pos = (center[0] + dist, center[1], center[2])
        per_cam = (per_pos, tuple(p_center.tolist()), (0.0, 0.0, 1.0))
        top_cam = (top_pos, tuple(center.tolist()),   (0.0, 1.0, 0.0))
        sid_cam = (sid_pos, tuple(s_center.tolist()), (0.0, 0.0, 1.0))

        # Build agent glyphs. Point flat glyphs ("billboards") towards camera.
        flat_ori        = lambda cp : np.asarray(cp)[None, :] - pos
        agnt_glyphs_0   = self.build_glyphs(
                            att_colors[:self.ts], 
                            pos[:self.ts], 
                            ori[:self.ts], 
                            self.body_rad, 
                            'cone')
        agnt_glyphs_1   = self.build_glyphs(
                            att_colors[self.ts:], 
                            pos[self.ts:], 
                            ori[self.ts:], 
                            self.body_rad, 
                            'sphere' if (DIFF_GEOM and not ONE_SWARM) else 'cone')
        pgon_glyphs = None
        if self.num_pgon > 0:
            pgon_colors = np.tile(clr.to_rgb(PGON_CLR), (self.num_pgon, 1)).astype(np.float32)
            pgon_colors = np.where(pg_done[:, None], np.asarray(clr.to_rgb(DONE_CLR), dtype=np.float32)[None, :], pgon_colors)
            pgon_ori = np.where(np.linalg.norm(pg_vel, axis=-1, keepdims=True) > 1e-6, pg_vel, np.array([1.0, 0.0, 0.0], dtype=np.float32)[None, :])
            pgon_scale = np.full((self.num_pgon,), self.body_rad * 0.5, dtype=np.float32)
            pgon_scale = np.where(pg_done, 0.0, pgon_scale)
            pgon_glyphs = self.build_glyphs(pgon_colors, pg_pos, pgon_ori, pgon_scale, 'sphere')
        view_glyphs_per = self.build_glyphs(
            colors, pos, flat_ori(per_pos), deat_fade * self.view_rad)
        view_glyphs_top = self.build_glyphs(
            colors, pos, flat_ori(top_pos), deat_fade * self.view_rad)
        view_glyphs_sid = self.build_glyphs(
            colors, pos, flat_ori(sid_pos), deat_fade * self.view_rad)

        # Render three views + info panel + exploration map.
        per_img = self._render_view(
            terrain,
            pos, 
            agnt_glyphs_0,
            agnt_glyphs_1,
            pgon_glyphs,
            view_glyphs_per, 
            hist,
            att_colors, 
            deat_fade, 
            (h, h), 
            view='perspective', 
            camera_pos=per_cam,
            transition=state.transition,
            clip_bounds=False,
        )
        top_img_terrain = self._render_view(
            terrain, 
            pos, 
            agnt_glyphs_0,
            agnt_glyphs_1, 
            pgon_glyphs,
            view_glyphs_top, 
            hist,
            att_colors,
            deat_fade, 
            (h//2, h//2), 
            view='top', 
            camera_pos=top_cam,
            transition=state.transition,
            toggle_agent_glyphs=False,
            toggle_view_glyphs=False,
            toggle_agent_lights=False,
            toggle_labels=False,
        )
        top_img_agents = self._render_view(
            None, 
            pos, 
            agnt_glyphs_0,
            agnt_glyphs_1, 
            pgon_glyphs,
            view_glyphs_top, 
            hist,
            att_colors,
            deat_fade, 
            (h//2, h//2), 
            view='top', 
            camera_pos=top_cam,
            transition=state.transition,
            background=False,
            toggle_bounds=False,
            toggle_terrain=False,
            toggle_trails=False,
        )
        sid_img = self._render_view(
            terrain, 
            pos, 
            agnt_glyphs_0,
            agnt_glyphs_1, 
            pgon_glyphs,
            view_glyphs_sid, 
            hist,
            att_colors, 
            deat_fade, 
            (h//2, h//2), 
            view='side', 
            camera_pos=sid_cam, 
            transition=state.transition,
        )
        inf_img = self._render_info_panel((int(0.2778 * h), h), state, att_colors, step_str)
        exp_img = self._render_expl_map((h//2, h//2), state)

        # Create top_image by compositing terrain, exploration map, and agent renders.
        top_img = np.maximum(exp_img, top_img_terrain) * 0.65 + top_img_terrain * 0.35
        top_img = np.where(top_img_agents == 0, top_img, top_img_agents)

        # Draw titles and borders.
        per_img = self.draw_title_and_border(per_img, 'PERSPECTIVE')
        top_img = self.draw_title_and_border(top_img, 'TOP ORTHOGONAL')
        sid_img = self.draw_title_and_border(sid_img, 'SIDE ORTHOGONAL')
        inf_img = self.draw_title_and_border(inf_img, 'INFORMATION', thick=2)

        # Stack and concatenate views. Scale to ensure equal heights.
        ort_img = np.vstack([top_img, sid_img])
        scale_h = lambda i, h : np.array(Image.fromarray(i).resize((i.shape[1], h)))
        per_img = scale_h(per_img, h)
        ort_img = scale_h(ort_img, h)
        inf_img = scale_h(inf_img, h)
        return np.concatenate([per_img, ort_img, inf_img], axis=1).astype(np.uint8)

# PERLIN2D ----------------------------------------------------------------- #

@jax.jit
def _interpolant(t: jnp.ndarray) -> jnp.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

# Return one octave of Perlin noise in [-1,1] for a size*size grid.
@partial(jax.jit, static_argnames=("size", "res"))
def _perlin2d(key, size: int, res: int):
    # Generate gradients.
    cell = size // res
    ang = jax.random.uniform(key, (res + 1, res + 1)) * (2 * math.pi)
    g = jnp.stack([jnp.cos(ang), jnp.sin(ang)], -1)

    # Gather corner gradients.
    y, x = jnp.mgrid[:size, :size]
    y = y / cell
    x = x / cell
    yi = jnp.floor(y).astype(jnp.int32)
    xi = jnp.floor(x).astype(jnp.int32)
    yf = y - yi
    xf = x - xi
    g00 = g[yi, xi]
    g10 = g[yi, xi + 1]
    g01 = g[yi + 1, xi]
    g11 = g[yi + 1, xi + 1]

    # Find dot products and interpolate.
    def dot(gv, dx, dy):
        return jnp.sum(gv * jnp.stack([dx, dy], -1), -1)
    n00 = dot(g00, xf, yf)
    n10 = dot(g10, xf - 1, yf)
    n01 = dot(g01, xf, yf - 1)
    n11 = dot(g11, xf - 1, yf - 1)
    u = _interpolant(xf)
    v = _interpolant(yf)
    nx0 = n00 * (1 - u) + n10 * u
    nx1 = n01 * (1 - u) + n11 * u
    nxy = nx0 * (1 - v) + nx1 * v
    return nxy * jnp.sqrt(2.0)

# Generate points of interest (POIs) and use to create
# circular levelled regions in the terrain heightmap.
def level_hmap(key, hmap, num_poi, rad) -> chex.Array:
    # Randomly initialise POIs.
    r    = jnp.asarray(rad, dtype=hmap.dtype)
    r_px = int(math.ceil(float(rad)))
    s    = 2 * r_px + 1
    pois = jax.random.randint(key, (num_poi, 2), 
                minval=r_px + 1, maxval=hmap.shape[-1] - r_px - 1)

    # Create circular mask.
    offs = jnp.arange(-r_px, r_px + 1, dtype=hmap.dtype)
    dy   = offs.reshape(s, 1)
    dx   = offs.reshape(1, s)
    mask = (dx * dx + dy * dy) <= (r * r) # bool, shape (S, S)

    # Apply mask to slices of hmap centred on POIs.
    def level_slice(i, hm):
        px = pois[i, 0]
        py = pois[i, 1]
        x0 = px - r_px
        y0 = py - r_px
        sub = lax.dynamic_slice(hm, (0, y0, x0), (2, s, s))
        sub_hmap = jnp.where(mask, hm[0, py, px], sub[0])
        sub_roi  = jnp.where(mask, 1.0, sub[1])
        new_sub  = jnp.stack([sub_hmap, sub_roi], axis=0)
        return lax.dynamic_update_slice(hm, new_sub, (0, y0, x0))
    return lax.fori_loop(0, pois.shape[0], level_slice, hmap)

# Use octaves of _perlin2d to create detailed noise. Values in [0, clip_max].
# Layer 0 = hmap, while layer 1 is a mask for regions of interest.
def fractal2d(
        key, 
        size:        int   = 64,
        res:         int   = 4,
        octaves:     int   = 2,
        persistence: float = 0.4,
        lacunarity:  float = 2.0,
        clip_max:    float = 0.8,
        exp:         float = 1.8,
        lower:       float = 0.2,
        num_poi:     int   = 16,
        poi_rad:     float = 5.0
    ):
    total = jnp.zeros((size, size), PRE)
    amp, freq = 1.0, 1.0
    for _ in range(octaves):
        key, subk = jax.random.split(key)
        total += amp * _perlin2d(subk, size=size, res=int(res * freq))
        amp   *= persistence
        freq  *= lacunarity
    norm = jnp.sum(jnp.array([persistence ** i for i in range(octaves)], PRE))
    hmap = ((total / norm + 1.0) * 0.5) # [0, 1]
    hmap = jnp.clip(hmap ** exp - lower, min=0.0, max=clip_max)
    hmap = jnp.stack([hmap, jnp.zeros_like(hmap)], axis=0) # add ROI mask layer
    return level_hmap(key, hmap, num_poi, poi_rad)

# Accept coords in [0, 1] and bilinearly sample height [0, 1].
# TODO: EXPERIMENT WITH REPLACING THIS WITH NEAREST FOR EFFICIENCY
def sample_hmap(hmap, xy):
    size = hmap.shape[-1]
    ix = xy[:, 0] * (size - 1)
    iy = xy[:, 1] * (size - 1)
    x0 = jnp.floor(ix).astype(jnp.int32)
    y0 = jnp.floor(iy).astype(jnp.int32)
    x1 = jnp.clip(x0 + 1, 0, size - 1)
    y1 = jnp.clip(y0 + 1, 0, size - 1)
    sx = ix - x0.astype(PRE)
    sy = iy - y0.astype(PRE)
    h00, h10 = hmap[0, y0, x0], hmap[0, y0, x1]
    h01, h11 = hmap[0, y1, x0], hmap[0, y1, x1]
    i0 = h00 * (1 - sx) + h10 * sx
    i1 = h01 * (1 - sx) + h11 * sx
    return i0 * (1 - sy) + i1 * sy

# END SCRIPT --------------------------------------------------------------- #

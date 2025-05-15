# ma_waterworld.py: policy blended with JAX-MD swarming forces

import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
from flax.struct import dataclass
from jax_md import space

# Action constants
ACT_LEFT, ACT_RIGHT, ACT_UP, ACT_DOWN, ACT_NONE = range(5)
# Environment size
SCREEN_W, SCREEN_H = 512, 512
# Stability epsilon
_eps = 1e-6

@dataclass
class BubbleStatus:
    pos_x: jnp.ndarray
    pos_y: jnp.ndarray
    vel_x: jnp.ndarray
    vel_y: jnp.ndarray
    bubble_type: jnp.int32
    valid: jnp.bool_
    poison_cnt: jnp.int32

# ─── Swarming potentials hyperparameters ─────────────────────────────────
J_align, D_align, a_align   = 1.0, 45.0, 3.0
J_avoid, D_avoid, a_avoid   = 25.0, 30.0, 3.0
J_cohesion, D_cohesion      = 0.1, 40.0

disp_fn, _ = space.free()

# pairwise displacement function
def pairwise_disp(R):
    return space.map_product(disp_fn)(R, R)

# total energy: alignment, avoidance, quadratic cohesion
def total_energy(state):
    R  = state['positions']            # [N,2]
    θ  = state['headings']             # [N]
    N  = jnp.stack([jnp.cos(θ), jnp.sin(θ)], axis=-1)  # [N,2]

    dR    = pairwise_disp(R)                          # [N,N,2]
    dr    = jnp.sqrt(jnp.sum(dR**2, axis=-1) + _eps**2)
    dotNN = jnp.clip(N @ N.T, -1.0, 1.0)

    # alignment energy: uses stop_gradient to avoid positional forces
    inside_A   = (dr < D_align) & (dr > _eps)
    dr_align   = lax.stop_gradient(dr)
    dot_align  = lax.stop_gradient(dotNN)
    wA         = jnp.where(inside_A, 1.0 - dr_align / D_align, 0.0)
    E_align    = jnp.where(
        inside_A,
        (J_align / a_align) * jnp.power(wA + _eps, a_align) * (1.0 - dot_align)**2,
        0.0
    )

    # avoidance energy: finite-range, power-law
    inside_R = dr < D_avoid
    wR       = jnp.where(inside_R, 1.0 - dr / D_avoid, 0.0)
    E_avoid  = jnp.where(
        inside_R,
        (J_avoid / a_avoid) * jnp.power(wR + _eps, a_avoid),
        0.0
    )

    # cohesion energy: simple spring within cutoff
    mask = dr < D_cohesion                   # boolean mask
    N_agents = R.shape[0]
    mask = mask.astype(jnp.float32) * (1.0 - jnp.eye(N_agents))  # remove self
    E_cohesion = 0.5 * J_cohesion * jnp.sum(mask * dr**2)

    return 0.5 * jnp.sum(E_align + E_avoid) + E_cohesion
    

# force generator via manual gradient
@jit
def force_fn(R, θ):
    return -grad(lambda R_, θ_: total_energy({'positions': R_, 'headings': θ_}), argnums=0)(R, θ)

# compute forces for all agents
def compute_swarm_forces(agent_states):
    R = jnp.stack([agent_states.pos_x, agent_states.pos_y], axis=-1)
    θ = jnp.arctan2(agent_states.vel_y, agent_states.vel_x + _eps)
    return force_fn(R, θ)  # [N,2]

# apply policy update blended with swarm force
@jax.vmap
def update_agent_state_swarm(agent: BubbleStatus,
                              direction: jnp.int32,
                              force: jnp.ndarray,
                              policy_weight: float=1.0,
                              energy_weight: float=0.1) -> BubbleStatus:
    # discrete policy velocity update
    vx = jnp.where(direction == ACT_RIGHT, agent.vel_x + 1, agent.vel_x)
    vx = jnp.where(direction == ACT_LEFT,  vx - 1, vx) * 0.95
    vy = jnp.where(direction == ACT_UP,   agent.vel_y - 1, agent.vel_y)
    vy = jnp.where(direction == ACT_DOWN, agent.vel_y + 1, vy) * 0.95

    # blend in JAX-MD force
    fx, fy = force
    vx = policy_weight * vx + energy_weight * fx
    vy = policy_weight * vy + energy_weight * fy

    # integrate and handle walls
    px = jnp.clip(agent.pos_x + vx, 1, SCREEN_W - 1)
    py = jnp.clip(agent.pos_y + vy, 1, SCREEN_H - 1)
    vx = jnp.where((px == 1) | (px == SCREEN_W - 1), 0, vx)
    vy = jnp.where((py == 1) | (py == SCREEN_H - 1), 0, vy)

    return BubbleStatus(pos_x=px, pos_y=py,
                         vel_x=vx, vel_y=vy,
                         bubble_type=agent.bubble_type,
                         valid=agent.valid,
                         poison_cnt=agent.poison_cnt)

# environment step: compute forces and update agents
def step_fn(state, directions):
    forces = compute_swarm_forces(state.agent_state)
    agents = update_agent_state_swarm(state.agent_state, directions, forces)
    return state.replace(agent_state=agents)

# ma_waterworld.py: swarming (avoidance+cohesion) + DSR propagation

import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
from flax.struct import dataclass
from jax_md import space

# Actions
LEFT, RIGHT, UP, DOWN, NONE = 0, 1, 2, 3, 4
# Environment constants
dt = 1.0
SCREEN_W, SCREEN_H = 512, 512
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
    info: jnp.ndarray
    info_prev: jnp.ndarray
    delta_prev: jnp.ndarray

# Hyperparameters
# Positional forces: avoidance + cohesion
a_avoid, D_avoid = 3.0, 30.0
J_avoid = 25.0
J_cohesion, D_cohesion = 0.1, 40.0
# DSR parameters
gamma, beta1, beta2 = 1.0, 0.9, 0.99
# Damping and speed cap
damping, max_speed = 0.1, 5.0

# Pairwise displacement (free space)
disp_fn, _ = space.free()
def pairwise_disp(R):
    return space.map_product(disp_fn)(R, R)

# Positional energy: avoidance + cohesion
def total_energy_pos(state):
    R = state['positions']                      # [N,2]
    dR = pairwise_disp(R)                       # [N,N,2]
    dr = jnp.sqrt(jnp.sum(dR**2, axis=-1) + _eps**2)  # [N,N]

    # avoidance: finite-range spring
    mask_a = (dr < D_avoid) & (dr > _eps)
    E_avoid = 0.5 * J_avoid * jnp.sum(mask_a * (dr**2))

    # cohesion: finite-range spring
    mask_c = (dr < D_cohesion) & (dr > _eps)
    E_cohesion = 0.5 * J_cohesion * jnp.sum(mask_c * (dr**2))

    return E_avoid + E_cohesion

# Compute swarm forces
@jit
def compute_swarm_forces(state):
    R = jnp.stack([state.pos_x, state.pos_y], axis=-1)
    θ = jnp.arctan2(state.vel_y, state.vel_x + _eps)
    # gradient of positional energy w.r.t. positions
    force_fn = lambda R_, θ_: -grad(
        lambda RR, TT: total_energy_pos({'positions': RR, 'headings': TT}),
        argnums=0)(R_, θ_)
    F = force_fn(R, θ)
    return F

# DSR information update
def compute_info_dsr(state):
    I, I0, d0 = state.info, state.info_prev, state.delta_prev
    R = jnp.stack([state.pos_x, state.pos_y], axis=-1)
    dR = pairwise_disp(R)
    dr = jnp.linalg.norm(dR, axis=-1)
    wC = jnp.where(dr < D_cohesion, (D_cohesion - dr) / D_cohesion, 0.0)
    wC = wC * (1.0 - jnp.eye(wC.shape[0]))

    sum_wC = jnp.sum(wC, axis=1)
    sum_Ij = jnp.einsum('ij,j->i', wC, I)
    Δ = sum_wC * I - sum_Ij

    I_dot = -gamma * Δ - beta1 * (Δ - d0) + beta2 * (I - I0)
    I_next = I + dt * I_dot
    return I_next, I, Δ

# Agent update: policy + swarm + damping + speed cap
def clamp_speed(vx, vy):
    speed = jnp.hypot(vx, vy)
    factor = jnp.minimum(1.0, max_speed / (speed + _eps))
    return vx * factor, vy * factor

@jax.vmap
def update_agent(agent: BubbleStatus, direction: jnp.int32, force: jnp.ndarray):
    # discrete policy
    vx = agent.vel_x + (direction == RIGHT) - (direction == LEFT)
    vy = agent.vel_y + (direction == DOWN) - (direction == UP)
    vx, vy = vx * 0.95, vy * 0.95

    # add swarm force
    fx, fy = force
    vx += fx; vy += fy

    # damping
    vx *= (1 - damping); vy *= (1 - damping)
    # speed cap
    vx, vy = clamp_speed(vx, vy)

    # integrate and bounds
    px = jnp.clip(agent.pos_x + vx, 0.0, SCREEN_W)
    py = jnp.clip(agent.pos_y + vy, 0.0, SCREEN_H)

    return BubbleStatus(
        pos_x=px, pos_y=py,
        vel_x=vx, vel_y=vy,
        bubble_type=agent.bubble_type,
        valid=agent.valid,
        poison_cnt=agent.poison_cnt,
        info=agent.info,
        info_prev=agent.info_prev,
        delta_prev=agent.delta_prev
    )

# Environment step

def step_fn(state, directions):
    forces = compute_swarm_forces(state)
    agents = update_agent(state.agent_state, directions, forces)
    I_next, I0, Δ = compute_info_dsr(state.agent_state)
    # update info fields
    agents = agents.replace(info=I_next, info_prev=I0, delta_prev=Δ)
    return state.replace(agent_state=agents)

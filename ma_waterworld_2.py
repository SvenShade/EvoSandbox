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

# Pairwise displacement (free-space)
disp_fn, _ = space.free()
def pairwise_disp(A, B):
    return space.map_product(disp_fn)(A, B)

# Total positional energy with per-agent coefficients
def total_energy_pos(R, items_R, item_types,
                     J_avoid_arr, D_avoid_arr,
                     J_coh_arr, D_coh_arr,
                     J_food_arr, D_food_arr,
                     J_poison_arr, D_poison_arr,
                     J_perim_arr, perim_rad_arr,
                     origins):
    N = R.shape[0]
    # agent-agent
    dR_aa = pairwise_disp(R, R) # [N,N,2]
    dr_aa = jnp.sqrt(jnp.sum(dR_aa**2, axis=-1) + _eps**2) # [N,N]
    da = D_avoid_arr[:,None] - dr_aa                      # [N,N]
    mask_a = dr_aa < D_avoid_arr[:,None]
    E_avoid = 0.5 * jnp.sum(J_avoid_arr[:,None] * mask_a * da**2)
    dc = dr_aa                                      # reuse dr_aa
    mask_c = dc < D_coh_arr[:,None]
    E_cohesion = 0.5 * jnp.sum(J_coh_arr[:,None] * mask_c * dc**2)

        # agent-item interactions
    dR_ai = pairwise_disp(R, items_R)                   # [N,M,2]
    dr_ai = jnp.sqrt(jnp.sum(dR_ai**2, axis=-1) + _eps**2)  # [N,M]
    # per-agent and per-item cutoffs
    Jf = J_food_arr[:, None]  # [N,1], broadcast to [N,M]
    Df = D_food_arr[:, None]  # [N,1]
    mask_f = dr_ai < Df       # [N,M]
    E_food = 0.5 * jnp.sum(Jf * mask_f * dr_ai**2)

    Jp = J_poison_arr[:, None]  # [N,1]
    Dp = D_poison_arr[:, None]  # [N,1]
    dp = Dp - dr_ai             # [N,M]
    mask_p = dr_ai < Dp         # [N,M]
    E_poison = 0.5 * jnp.sum(Jp * mask_p * dp**2)

    # perimeter
    origins = origins.reshape(N,2)
    dR0 = R - origins                            # [N,2]
    dr0 = jnp.sqrt(jnp.sum(dR0**2, axis=-1) + _eps**2) # [N]
    pr = perim_rad_arr
    dr0_c = jnp.minimum(dr0, pr)
    E_perim = 0.5 * jnp.sum(J_perim_arr * dr0_c**2)

    return E_avoid + E_cohesion + E_food + E_poison + E_perim

# Compute swarm forces with per-agent coefficients
def compute_swarm_forces(state):
    # agent positions
    R = jnp.stack([state.agent_state.pos_x, state.agent_state.pos_y], axis=-1)
    # items
    items_R = jnp.stack([state.items.pos_x, state.items.pos_y], axis=-1)
    item_types = state.items.bubble_type
    # per-agent arrays
    ba = state.agent_state
    J_avoid_arr   = ba.J_avoid
    D_avoid_arr   = ba.D_avoid
    J_coh_arr     = ba.J_cohesion
    D_coh_arr     = ba.D_cohesion
    J_food_arr    = ba.J_food
    D_food_arr    = ba.D_food
    J_poison_arr  = ba.J_poison
    D_poison_arr  = ba.D_poison
    J_perim_arr   = ba.J_perimeter
    perim_rad_arr = ba.perimeter_radius
    origins       = jnp.stack([ba.origin_x, ba.origin_y], axis=-1)

    energy_R = functools.partial(
        total_energy_pos,
        items_R=items_R,
        item_types=item_types,
        J_avoid_arr=J_avoid_arr,
        D_avoid_arr=D_avoid_arr,
        J_coh_arr=J_coh_arr,
        D_coh_arr=D_coh_arr,
        J_food_arr=J_food_arr,
        D_food_arr=D_food_arr,
        J_poison_arr=J_poison_arr,
        D_poison_arr=D_poison_arr,
        J_perim_arr=J_perim_arr,
        perim_rad_arr=perim_rad_arr,
        origins=origins
    )
    F = -grad(energy_R)(R)
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

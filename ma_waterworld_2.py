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

# Total positional energy: agents, food, poison
def total_energy_pos(R, food_pos, poison_pos):
    # agent-agent springs (repulsion + cohesion)
    dR_aa = pairwise_disp(R, R)  # [N,N,2]
    dr_aa = jnp.linalg.norm(dR_aa, axis=-1) + _eps  # [N,N]
    mask_avoid = dr_aa < D_avoid
    da = D_avoid - dr_aa
    E_avoid = 0.5 * J_avoid * jnp.sum(mask_avoid * da**2)
    mask_coh = dr_aa < D_cohesion
    E_cohesion = 0.5 * J_cohesion * jnp.sum(mask_coh * dr_aa**2)

    # agent-food attraction
    dR_af = pairwise_disp(R, food_pos)  # [N,M,2]
    dr_af = jnp.linalg.norm(dR_af, axis=-1) + _eps
    mask_food = dr_af < D_food
    E_food = 0.5 * J_food * jnp.sum(mask_food * dr_af**2)

    # agent-poison repulsion
    dR_ap = pairwise_disp(R, poison_pos)  # [N,K,2]
    dr_ap = jnp.linalg.norm(dR_ap, axis=-1) + _eps
    mask_p = dr_ap < D_poison
    dp = D_poison - dr_ap
    E_poison = 0.5 * J_poison * jnp.sum(mask_p * dp**2)

    return E_avoid + E_cohesion + E_food + E_poison

# Compute swarm forces via gradient of energy
def compute_swarm_forces(state):
    # agent positions
    R = jnp.stack([state.agent_state.pos_x, state.agent_state.pos_y], axis=-1)
    # extract item positions by type
    items = state.items
    # constants for item types (assumed defined elsewhere)
    PREY = 1       # food type
    OBSTACLE = 2   # poison type
    # select food positions
    food_mask = items.bubble_type == PREY
    food_pos = jnp.stack([items.pos_x[food_mask], items.pos_y[food_mask]], axis=-1)
    # select poison positions
    poison_mask = items.bubble_type == OBSTACLE
    poison_pos = jnp.stack([items.pos_x[poison_mask], items.pos_y[poison_mask]], axis=-1)

    # bind food and poison arrays into energy function
    energy_R = functools.partial(total_energy_pos,
                                 food_pos=food_pos,
                                 poison_pos=poison_pos)
    # compute forces: negative gradient w.r.t. R
    F = -grad(energy_R)(R)
    return jit(lambda x: F)(state)  # return forces for each agent

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

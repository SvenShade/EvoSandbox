import jax
import jax.numpy as jnp
from jax import jit, vmap
from flax.struct import dataclass
from jax_md import space, quantity

# Action constants
ACT_LEFT, ACT_RIGHT, ACT_UP, ACT_DOWN, ACT_NONE = range(5)
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


def safe_normalize(x, axis=-1, eps=_eps):
    norm = jnp.linalg.norm(x, axis=axis, keepdims=True)
    norm_safe = jnp.where(norm < eps, 1.0, norm)
    return x / norm_safe


J_align, D_align, a_align = 1.0, 45.0, 3.0
J_avoid, D_avoid, a_avoid = 25.0, 30.0, 3.0
J_cohesion, D_cohesion = 0.1, 40.0

disp_fn, _ = space.free()
def pairwise_disp(R):
    return space.map_product(disp_fn)(R, R)


def total_energy(state):
    R = state['positions']              # [N,2]
    θ = state['headings']               # [N]
    N = jnp.stack([jnp.cos(θ), jnp.sin(θ)], axis=-1)  # [N,2]

    # pairwise info
    dR = pairwise_disp(R)              # [N,N,2]
    dr = jnp.sqrt(jnp.sum(dR**2, axis=-1) + _eps**2)  # [N,N]
    dotNN = jnp.clip(N @ N.T, -1.0, 1.0)

    # Alignment
    wA = jnp.clip(1.0 - dr / D_align, 0.0, 1.0)
    E_align = (J_align / a_align) * jnp.power(wA + _eps, a_align) * (1.0 - dotNN)**2

    # Avoidance
    wR = jnp.clip(1.0 - dr / D_avoid, 0.0, 1.0)
    E_avoid = (J_avoid / a_avoid) * jnp.power(wR + _eps, a_avoid)

    # Cohesion: quadratic attraction to neighbors within radius
    wC = jnp.clip((D_cohesion - dr) / D_cohesion, 0.0, 1.0)       # [N,N]
    # zero-out self-interaction
    N_agents = R.shape[0]
    wC = wC * (1.0 - jnp.eye(N_agents))                          # no self-term
    E_cohesion = 0.5 * J_cohesion * jnp.sum(wC * dr**2)

    # total energy: half for pairwise, full for cohesion
    return 0.5 * jnp.sum(E_align + E_avoid) + jnp.sum(E_cohesion)


def force_fn_manual(R, θ):
    # negative gradient w.r.t. positions
    return -grad(lambda R_, θ_: total_energy({'positions': R_, 'headings': θ_}), argnums=0)(R, θ)


force_fn = jit(force_fn_manual)
def compute_swarm_forces(agent_states):
    R = jnp.stack([agent_states.pos_x, agent_states.pos_y], axis=-1)   # [N,2]
    θ = jnp.arctan2(agent_states.vel_y, agent_states.vel_x + _eps)     # [N]

    # Debug: inspect energy and raw gradient for small test
    E_sample = total_energy({'positions': R, 'headings': θ})
    grad_R = grad(lambda R_: total_energy({'positions': R_, 'headings': θ}), argnums=0)(R)
    print("Debug -- E_sample:", E_sample)
    print("Debug -- grad_R:", grad_R)

    # Compute forces
    forces = force_fn(R, θ)
    return forces                                                   # [N,2]


@jax.vmap
def update_agent_state_swarm(agent: BubbleStatus,
                              direction: jnp.int32,
                              force: jnp.ndarray,
                              policy_weight: float = 1.0,
                              energy_weight: float = 0.1) -> BubbleStatus:

    vx = agent.vel_x
    vx = jnp.where(direction == ACT_RIGHT, vx + 1, vx)
    vx = jnp.where(direction == ACT_LEFT,  vx - 1, vx)
    vx *= 0.95
    vy = agent.vel_y
    vy = jnp.where(direction == ACT_UP,   vy - 1, vy)
    vy = jnp.where(direction == ACT_DOWN, vy + 1, vy)
    vy *= 0.95

    fx, fy = force
    vx = policy_weight * vx + energy_weight * fx
    vy = policy_weight * vy + energy_weight * fy
    px = agent.pos_x + vx
    py = agent.pos_y + vy

    px = jnp.clip(px, 1, SCREEN_W - 1)
    py = jnp.clip(py, 1, SCREEN_H - 1)
    vx = jnp.where((px == 1) | (px == SCREEN_W - 1), 0, vx)
    vy = jnp.where((py == 1) | (py == SCREEN_H - 1), 0, vy)

    return BubbleStatus(
        pos_x=px, pos_y=py,
        vel_x=vx, vel_y=vy,
        bubble_type=agent.bubble_type,
        valid=agent.valid,
        poison_cnt=agent.poison_cnt
    )


def step_fn(state, directions):
    forces = compute_swarm_forces(state.agent_state)
    agents = update_agent_state_swarm(
        state.agent_state,
        directions,
        forces,
        policy_weight=1.0,
        energy_weight=0.1
    )
    # ... rest of environment logic (food, poison, rewards, observations)
    state = state.replace(agent_state=agents)
    return state

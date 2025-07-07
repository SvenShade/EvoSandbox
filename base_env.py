import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass, field

# --- Perlin Noise Utilities ---

def interpolant(t: jnp.ndarray) -> jnp.ndarray:
    """
    Smoothstep interpolation function for Perlin noise.
    """
    return t**3 * (t * (t * 6 - 15) + 10)


def generate_perlin_noise_2d(
    key: jnp.ndarray,
    shape: tuple[int, int],
    res: tuple[int, int],
    tileable: tuple[bool, bool] = (False, False)
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate single-octave Perlin noise heightmap using JAX RNG key.
    Returns (next_key, heightmap).
    """
    key, subkey = jax.random.split(key)
    angles = jax.random.uniform(subkey, (res[0] + 1, res[1] + 1), 0.0, 2*jnp.pi)
    gradients = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
    if tileable[0]: gradients = gradients.at[-1, :].set(gradients[0, :])
    if tileable[1]: gradients = gradients.at[:, -1].set(gradients[:, 0])

    grid = jnp.stack(jnp.meshgrid(
        jnp.linspace(0, res[0], shape[0], endpoint=False) % 1,
        jnp.linspace(0, res[1], shape[1], endpoint=False) % 1,
        indexing='ij'
    ), axis=-1)

    d0, d1 = shape[0] // res[0], shape[1] // res[1]
    grads = jnp.repeat(jnp.repeat(gradients, d0, axis=0), d1, axis=1)
    g00 = grads[:-d0, :-d1]
    g10 = grads[d0:, :-d1]
    g01 = grads[:-d0, d1:]
    g11 = grads[d0:, d1:]

    n00 = jnp.sum(grid * g00, axis=-1)
    n10 = jnp.sum((grid - jnp.array([1.0, 0.0])) * g10, axis=-1)
    n01 = jnp.sum((grid - jnp.array([0.0, 1.0])) * g01, axis=-1)
    n11 = jnp.sum((grid - jnp.array([1.0, 1.0])) * g11, axis=-1)

    t = interpolant(grid)
    n0 = n00 * (1 - t[:,:,0]) + t[:,:,0] * n10
    n1 = n01 * (1 - t[:,:,0]) + t[:,:,0] * n11
    noise = jnp.sqrt(2) * ((1 - t[:,:,1]) * n0 + t[:,:,1] * n1)
    return key, noise


def generate_fractal_noise_2d(
    key: jnp.ndarray,
    shape: tuple[int,int],
    res: tuple[int,int],
    octaves: int = 4,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    tileable: tuple[bool,bool] = (False, False)
) -> tuple[jnp.ndarray,jnp.ndarray]:
    """
    Generate multi-octave Perlin noise; returns (next_key, heightmap).
    """
    noise = jnp.zeros(shape)
    frequency = 1.0
    amplitude = 1.0
    for _ in range(octaves):
        key, single = generate_perlin_noise_2d(
            key,
            shape,
            (int(res[0]*frequency), int(res[1]*frequency)),
            tileable
        )
        noise += amplitude * single
        frequency *= lacunarity
        amplitude *= persistence
    return key, noise


def interpolate_heightmap(
    heightmap: jnp.ndarray,
    xy: jnp.ndarray,
    world_size: tuple[float, float]
) -> jnp.ndarray:
    """
    Bilinear interpolation of heightmap at XY positions.
    """
    H, W = heightmap.shape
    wx, wy = world_size
    grid_x = (xy[:,0] + wx/2) * (W-1) / wx
    grid_y = (xy[:,1] + wy/2) * (H-1) / wy
    x0 = jnp.floor(grid_x).astype(jnp.int32)
    y0 = jnp.floor(grid_y).astype(jnp.int32)
    x1 = jnp.clip(x0+1, 0, W-1)
    y1 = jnp.clip(y0+1, 0, H-1)
    wx_f = grid_x - x0
    wy_f = grid_y - y0
    h00 = heightmap[y0, x0]
    h10 = heightmap[y0, x1]
    h01 = heightmap[y1, x0]
    h11 = heightmap[y1, x1]
    return (
        h00*(1-wx_f)*(1-wy_f) +
        h10*wx_f*(1-wy_f) +
        h01*(1-wx_f)*wy_f +
        h11*wx_f*wy_f
    )

# --- Environment Definition ---

@dataclass
class Params:
    # Core parameters
    num_drones: int = 10
    num_teams: int = 2
    world_radius: float = 1.0
    max_acceleration: float = 0.01
    min_speed: float = 0.0
    max_speed: float = 0.03
    collision_radius: float = 0.05
    view_radius: float = 0.2
    num_poi: int = 5
    ground_res: int = 16
    num_ent: int = 5
    max_steps: int = 200
    # Perlin params
    map_size: int = 256
    perlin_octaves: int = 4
    # Precomputed squares
    world_radius_sq: float = field(init=False)
    view_radius_sq: float = field(init=False)
    collision_radius_sq: float = field(init=False)

    def __post_init__(self):
        self.world_radius_sq     = self.world_radius**2
        self.view_radius_sq      = self.view_radius**2
        self.collision_radius_sq = self.collision_radius**2
        # number of cells per axis for spatial hash
        self.num_cells = jnp.ceil(2*self.world_radius / self.view_radius).astype(int)
        self.num_cells_sq = self.num_cells**2

# Precompute ground grid offsets
_lin = jnp.linspace; _mesh = jnp.meshgrid
GRID_OFFSETS = _mesh(
    _lin(-1,1,Params.ground_res)*Params.view_radius,
    _lin(-1,1,Params.ground_res)*Params.view_radius,
    indexing='xy'
)
GRID_OFFSETS = jnp.stack([GRID_OFFSETS[0].ravel(), GRID_OFFSETS[1].ravel()], -1)



@jax.jit

def build_cell_list(pos_xy: jnp.ndarray, params: Params):
    """
    Compute cell_id and cell index structures for N agents.
    Returns:
      order: (N,) indices sorted by cell_id
      starts: (C*C+1,) start indices in order for each cell
      cell_id: (N,) cell_id for each agent
    """
    # cell coords in [0, C)
    cell_xy = jnp.floor((pos_xy + params.world_radius) / params.view_radius).astype(int)
    cell_xy = jnp.clip(cell_xy, 0, params.num_cells-1)
    cell_id = cell_xy[:,0] * params.num_cells + cell_xy[:,1]
    # sort by cell_id
    order = jnp.argsort(cell_id)
    sorted_c = cell_id[order]
    # counts per cell
    counts = jnp.bincount(sorted_c, length=params.num_cells_sq)
    # prefix sums for start
    starts = jnp.concatenate([jnp.array([0]), jnp.cumsum(counts)])
    return order, starts, cell_id

# --- Helpers: neighbor lookup ---
@jax.jit

def gather_neighbors(order: jnp.ndarray, starts: jnp.ndarray, cell_id: jnp.ndarray, params: Params):
    """
    For each agent, gather indices of neighbors in adjacent cells.
    Returns ragged array padded to max_k agents: shape (N, K)
    """
    N = cell_id.shape[0]
    C = params.num_cells
    # compute each agent's cell coords
    cell_xy = jnp.stack([cell_id // C, cell_id % C], axis=-1)
    # offsets for 3x3
    offs = jnp.array([[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]])
    def per_agent(i, carry):
        ci, cj = cell_xy[i]
        neigh_ids = []
        for off in offs:
            ni, nj = ci+off[0], cj+off[1]
            # check bounds
            if (ni>=0) & (ni<C) & (nj>=0)&(nj<C):
                cid = ni*C + nj
                start = starts[cid]
                end = starts[cid+1]
                neigh_ids.append(order[start:end])
        alln = jnp.concatenate(neigh_ids)
        # pad/truncate to K=params.num_ent*3
        K = params.num_ent * 3
        pad = K - alln.shape[0]
        def pad_fn(): return jnp.concatenate([alln, jnp.zeros(pad, dtype=int)])
        def trunc(): return alln[:K]
        return jax.lax.cond(pad>0, pad_fn, trunc)
    # vmap
    neighbors = jax.vmap(per_agent, in_axes=(0,None))(jnp.arange(N), None)
    return neighbors  # shape (N,K)

# --- Environment Functions ---
@jax.jit

def reset(key: jnp.ndarray, params: Params):
    # generate terrain, state as before
    # omitted for brevity...
    pass

@jax.jit

def step(state, action, params: Params):
    pos, vel, alive, teams, poi, heightmap, sc, key = state
    # dynamics and collisions as before except neighbor collision:
    # build 2D positions
    pos_xy = pos[:, :2]
    order, starts, cell_id = build_cell_list(pos_xy, params)
    neigh = gather_neighbors(order, starts, cell_id, params)
    # scatter neighbors for each agent
    def collide_agent(i):
        idxs = neigh[i]
        rels = pos[i] - pos[idxs]
        d2s = jnp.sum(rels**2, axis=-1)
        return jnp.any(d2s < params.collision_radius_sq)
    collide = jax.vmap(collide_agent)(jnp.arange(params.num_drones))
    # proceed with alive update, etc.
    # ... continue rest of step ...
    pass

@jax.jit

def get_obs(state, params: Params):
    pos, vel, alive, teams, poi, heightmap, sc, key = state
    # build neighbor lists as in step
    pos_xy = pos[:, :2]
    order, starts, cell_id = build_cell_list(pos_xy, params)
    neigh = gather_neighbors(order, starts, cell_id, params)
    # for each agent, select top-E friend/enemy from its k neighbors
    # use vmap over agents similar to collision
    # ... implement per-agent selection using rels and argsort over k instead of N ...
    pass



@jax.jit

def reset(key: jnp.ndarray, params: Params):
    """
    JIT-compiled reset: generates a new Perlin map and initial state.
    Returns obs, state.
    """
    # RNG splits
    key, sk1, sk2 = jax.random.split(key, 3)
    # Generate heightmap via JAX
    key, heightmap = generate_fractal_noise_2d(
        sk1,
        (params.map_size, params.map_size),
        res=(8,8),
        octaves=params.perlin_octaves
    )
    # Sample positions and velocities
    pos = jax.random.uniform(sk2, (params.num_drones,3),
                             -params.world_radius, params.world_radius)
    vk = jax.random.PRNGKey(key[0].astype(jnp.int32))
    vel = jax.random.uniform(vk, (params.num_drones,3),
                             params.min_speed, params.max_speed)
    vnorm = jnp.linalg.norm(vel, axis=1, keepdims=True)
    vel = vel * (jnp.clip(vnorm, params.min_speed, params.max_speed)/(vnorm+1e-8))

    # Teams & POIs
    tk, xk = jax.random.split(key)
    teams = jax.random.randint(tk, (params.num_drones,), 0, params.num_teams)
    poi_xy = jax.random.uniform(xk, (params.num_poi,2),
                                -params.world_radius, params.world_radius)
    poi = jnp.concatenate([poi_xy,
                            interpolate_heightmap(
                                heightmap, poi_xy,
                                (2*params.world_radius,2*params.world_radius)
                            )[:,None]
                           ], axis=-1)

    alive = jnp.ones(params.num_drones, bool)
    state = (pos, vel, alive, teams, poi, heightmap, 0, key)
    obs = get_obs(state, params)
    return obs, state

@jax.jit
def step(state, action: jnp.ndarray, params: Params):
    """
    Advance environment one timestep.
    Returns obs, new_state, rewards, done, info.
    """
    pos, vel, alive, teams, poi, perlin_fn, step_count, key = state
    w2 = params.world_radius_sq
    c2 = params.collision_radius_sq

    # Integrate dynamics
    vel = vel + action * params.max_acceleration
    vnorm = jnp.linalg.norm(vel, axis=1, keepdims=True)
    vel = vel * (jnp.clip(vnorm, params.min_speed, params.max_speed) / (vnorm + 1e-8))
    pos = pos + vel

    # Out-of-bounds and terrain collision
    oob = jnp.sum(pos[:, :2]**2, axis=1) > w2
    land = pos[:, 2] < perlin_fn(pos[:, :2])

    # Drone-drone collision with spatial partitioning
    rel = pos[:, None, :] - pos[None, :, :]
    dist2 = jnp.sum(rel**2, axis=-1)
    cell = jnp.floor((pos[:, :2] + params.world_radius) / params.view_radius).astype(jnp.int32)
    cell_i = cell[:, None, :]
    cell_j = cell[None, :, :]
    dc = jnp.abs(cell_i - cell_j)
    nearby = (dc[:, :, 0] <= 1) & (dc[:, :, 1] <= 1)
    rel_part = rel * nearby[:, :, None]
    dist2_part = jnp.sum(rel_part**2, axis=-1)
    collide = jnp.any(dist2_part < c2, axis=1)

    alive = alive & ~oob & ~land & ~collide
    done = (step_count + 1 >= params.max_steps) | (~jnp.any(alive))

    new_state = (pos, vel, alive, teams, poi, perlin_fn, step_count + 1, key)
    obs, _ = get_obs(new_state, params), new_state
    return obs, new_state, jnp.zeros(params.num_drones), done, {}

@jax.jit
def get_obs(state, params: Params) -> jnp.ndarray:
    """
    Compute per-drone observation vectors.
    Outputs concatenated features:
      - Friendly rel-pos (max num_ent)
      - Enemy   rel-pos (max num_ent)
      - POI     rel-pos (max num_ent)
      - Center distance ratio (1)
      - Ground height patch (ground_res**2)
    """
    pos, _, _, teams, poi, perlin_fn, _, _ = state
    N, E = params.num_drones, params.num_ent
    v2 = params.view_radius_sq

    # Relative vectors and visibility mask
    rel = pos[:, None, :] - pos[None, :, :]
    dist2 = jnp.sum(rel**2, axis=-1)
    vis = dist2 <= v2

    # Spatial partition for drone neighbors
    cell = jnp.floor((pos[:, :2] + params.world_radius) / params.view_radius).astype(jnp.int32)
    cell_i = cell[:, None, :]
    cell_j = cell[None, :, :]
    dc = jnp.abs(cell_i - cell_j)
    nearby = (dc[:, :, 0] <= 1) & (dc[:, :, 1] <= 1)

    def select(mask_mat, data):
        mask_comb = mask_mat & nearby
        d2 = jnp.where(mask_comb, jnp.sum(data**2, axis=-1), jnp.inf)
        idx = jnp.argsort(d2, axis=1)[:, :E]
        return jnp.take_along_axis(data * mask_comb[:, :, None], idx[:, :, None], axis=1)

    ti, tj = teams[:, None], teams[None, :]
    rf = select((tj == ti) & vis, rel).reshape(N, -1)
    re = select((tj != ti) & vis, rel).reshape(N, -1)

    # POI neighbor selection (range-based)
    rpoi = poi[None, :, :] - pos[:, None, :]
    poi_vis = jnp.sum(rpoi**2, axis=-1) <= v2
    rp = select(poi_vis, rpoi).reshape(N, -1)

    # Distance from center
    cr = (jnp.linalg.norm(pos[:, :2], axis=1) / params.world_radius)[:, None]

    # Ground patch sampling + circular mask
    global_xy = pos[:, None, :2] + GRID_OFFSETS[None, :, :]
    d2g = jnp.sum((global_xy - pos[:, None, :2])**2, axis=-1)
    hg = perlin_fn(global_xy.reshape(-1, 2)).reshape(N, -1)
    go = jnp.where(d2g <= v2, hg, 0.0)

    return jnp.concatenate([rf, re, rp, cr, go], axis=-1)

@jax.jit
def get_obs_graph(state, params: Params) -> jnp.ndarray:
    """
    Build graph node features per drone: (num_ent*3, 4) tensor.
    Node = [dx, dy, dz, type_id], type_id in {0,1,2}
    """
    pos, _, _, teams, poi, perlin_fn, _, _ = state
    N, E = params.num_drones, params.num_ent
    v2 = params.view_radius_sq

    rel = pos[:, None, :] - pos[None, :, :]
    dist2 = jnp.sum(rel**2, axis=-1)
    vis = dist2 <= v2
    cell = jnp.floor((pos[:, :2] + params.world_radius) / params.view_radius).astype(jnp.int32)
    dc = jnp.abs(cell[:, None, :] - cell[None, :, :])
    nearby = (dc[:, :, 0] <= 1) & (dc[:, :, 1] <= 1)
    ti, tj = teams[:, None], teams[None, :]

    def make_nodes(mask_mat, data, t_id: float):
        mask_comb = mask_mat & nearby
        d2 = jnp.where(mask_comb, jnp.sum(data**2, axis=-1), jnp.inf)
        idx = jnp.argsort(d2, axis=1)[:, :E]
        nodes = jnp.take_along_axis(data * mask_comb[:, :, None], idx[:, :, None], axis=1)
        types = jnp.full((N, E, 1), t_id)
        return jnp.concatenate([nodes, types], axis=-1)

    nf = make_nodes((tj == ti) & vis, rel, 0.0)
    ne = make_nodes((tj != ti) & vis, rel, 1.0)
    np_ = make_nodes(jnp.sum((poi[None, :, :] - pos[:, None, :])**2, axis=-1) <= v2,
                      poi[None, :, :] - pos[:, None, :], 2.0)
    return jnp.concatenate([nf, ne, np_], axis=1)

# --- Integration with JaxMARL & Mava ---
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Box as JMBox

class SimpleTagEnv(MultiAgentEnv):
    """
    JaxMARL-compatible wrapper implementing MultiAgentEnv.
    Reset/step operate on dict-based agent observations.
    """
    def __init__(self, params: Params):
        self.params = params
        self.agents = [f"agent_{i}" for i in range(params.num_drones)]
        obs_dim = params.num_ent * 9 + 1 + params.ground_res**2
        self.observation_spaces = {a: JMBox(-jnp.inf, jnp.inf, (obs_dim,), jnp.float32)
                                   for a in self.agents}
        low, high = -params.max_acceleration, params.max_acceleration
        self.action_spaces = {a: JMBox(low, high, (3,), jnp.float32)
                              for a in self.agents}

    def reset(self, key: jnp.ndarray):
        obs_arr, state = reset(key, self.params)
        return {a: obs_arr[i] for i, a in enumerate(self.agents)}, state

    def step(self, key: jnp.ndarray, state: any, actions: dict):
        act_arr = jnp.stack([actions[a] for a in self.agents])
        obs_arr, new_state, rews, done, info = step(state, act_arr, self.params)
        obs = {a: obs_arr[i] for i, a in enumerate(self.agents)}
        rewards = {a: float(rews[i]) for i, a in enumerate(self.agents)}
        done_dict = {a: bool(done) for a in self.agents}
        return obs, new_state, rewards, done_dict, info

from jaxmarl.py import JaxMarlWrapper  # noqa: E402

def make_mava_env(params: Params):
    """
    Wraps SimpleTagEnv for Mava via JaxMarlWrapper.
    """
    env = SimpleTagEnv(params)
    return JaxMarlWrapper(env, has_global_state=False, time_limit=params.max_steps)

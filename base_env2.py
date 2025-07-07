import jax
import jax.numpy as jnp
from dataclasses import dataclass, field

# --- Environment Parameters ---
@dataclass
class Params:
    # Agent and team counts
    num_drones: int = 10              # Total number of drones
    num_teams: int = 2                # Number of teams (friendly vs adversarial)
    # World and motion settings
    world_radius: float = 1.0         # Radius of circular world (XY-plane)
    view_radius: float = 0.2          # Visibility radius for sensing
    collision_radius: float = 0.05    # Distance threshold for collisions
    max_acceleration: float = 0.01    # Maximum per-axis acceleration
    min_speed: float = 0.0            # Minimum speed after clamp
    max_speed: float = 0.03           # Maximum speed after clamp
    # Points of Interest and ground sampling
    num_poi: int = 5                  # Number of POIs randomly placed
    ground_res: int = 16              # Resolution of ground height patch
    # Observation limits
    num_ent: int = 5                  # Max entities per type (friendly/enemy/POI)
    max_steps: int = 200              # Max steps per episode
    # Terrain (Perlin noise) settings
    map_size: int = 256               # Heightmap resolution
    perlin_octaves: int = 4           # Number of fractal octaves
    # Precomputed squared thresholds for efficiency
    world_radius_sq: float = field(init=False)
    view_radius_sq: float = field(init=False)
    collision_radius_sq: float = field(init=False)
    # Spatial hashing grid
    num_cells: int = field(init=False)
    num_cells_sq: int = field(init=False)

    def __post_init__(self):
        # Precompute squared radii
        self.world_radius_sq     = self.world_radius ** 2
        self.view_radius_sq      = self.view_radius ** 2
        self.collision_radius_sq = self.collision_radius ** 2
        # Determine number of grid cells per axis
        self.num_cells    = int(jnp.ceil(2 * self.world_radius / self.view_radius))
        self.num_cells_sq = self.num_cells ** 2

# --- Perlin Noise Utilities ---

def interpolant(t: jnp.ndarray) -> jnp.ndarray:
    """
    Smoothstep interpolation function for Perlin noise.
    """
    return t**3 * (t * (t * 6 - 15) + 10)

@jax.jit
def generate_perlin_noise_2d(
    key: jnp.ndarray,
    shape: tuple[int, int],
    res: tuple[int, int],
    tileable: tuple[bool, bool] = (False, False)
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate single-octave Perlin noise heightmap using JAX RNG.
    Returns (next_key, noise) of shape `shape`.
    """
    # Split RNG for gradients
    key, subkey = jax.random.split(key)
    # Random gradient angles
    angles = jax.random.uniform(subkey, (res[0] + 1, res[1] + 1), 0.0, 2*jnp.pi)
    grads  = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
    # Enforce tileable edges
    if tileable[0]: grads = grads.at[-1, :].set(grads[0, :])
    if tileable[1]: grads = grads.at[:, -1].set(grads[:, 0])
    # Build fractional coordinate grid in [0,1)
    grid = jnp.stack(jnp.meshgrid(
        jnp.linspace(0, res[0], shape[0], endpoint=False) % 1,
        jnp.linspace(0, res[1], shape[1], endpoint=False) % 1,
        indexing='ij'
    ), axis=-1)
    # Tile gradients to full resolution
    d0, d1 = shape[0] // res[0], shape[1] // res[1]
    grads_t = jnp.repeat(jnp.repeat(grads, d0, axis=0), d1, axis=1)
    g00 = grads_t[:-d0, :-d1]
    g10 = grads_t[d0:, :-d1]
    g01 = grads_t[:-d0, d1:]
    g11 = grads_t[d0:, d1:]
    # Dot products (ramp values)
    n00 = jnp.sum(grid * g00, axis=-1)
    n10 = jnp.sum((grid - jnp.array([1.0, 0.0])) * g10, axis=-1)
    n01 = jnp.sum((grid - jnp.array([0.0, 1.0])) * g01, axis=-1)
    n11 = jnp.sum((grid - jnp.array([1.0, 1.0])) * g11, axis=-1)
    # Interpolation weights
    t   = interpolant(grid)
    n0  = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1  = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    noise = jnp.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)
    return key, noise

@jax.jit
def generate_fractal_noise_2d(
    key: jnp.ndarray,
    shape: tuple[int, int],
    res: tuple[int, int],
    octaves: int = 4,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    tileable: tuple[bool, bool] = (False, False)
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate fractal (multi-octave) Perlin noise heightmap.
    """
    noise = jnp.zeros(shape)
    frequency, amplitude = 1.0, 1.0
    for _ in range(octaves):
        key, single = generate_perlin_noise_2d(
            key,
            shape,
            (int(res[0] * frequency), int(res[1] * frequency)),
            tileable
        )
        noise += amplitude * single
        frequency *= lacunarity
        amplitude *= persistence
    return key, noise

@jax.jit
def interpolate_heightmap(
    heightmap: jnp.ndarray,
    xy: jnp.ndarray,
    world_size: tuple[float, float]
) -> jnp.ndarray:
    """
    Bilinearly interpolate a precomputed heightmap at points xy.
    """
    H, W = heightmap.shape
    wx, wy = world_size
    # Map xy from [-wx/2,wx/2] to pixel coords [0,W-1] and [0,H-1]
    gx = (xy[:, 0] + wx/2) * (W - 1) / wx
    gy = (xy[:, 1] + wy/2) * (H - 1) / wy
    x0 = jnp.floor(gx).astype(int); x1 = jnp.clip(x0 + 1, 0, W - 1)
    y0 = jnp.floor(gy).astype(int); y1 = jnp.clip(y0 + 1, 0, H - 1)
    fx, fy = gx - x0, gy - y0
    h00 = heightmap[y0, x0]; h10 = heightmap[y0, x1]
    h01 = heightmap[y1, x0]; h11 = heightmap[y1, x1]
    return (
        h00 * (1 - fx) * (1 - fy)
      + h10 * fx       * (1 - fy)
      + h01 * (1 - fx) * fy
      + h11 * fx       * fy
    )

class Perlin2D:
    """
    Precomputes a fractal Perlin heightmap and provides callable sampling.
    """
    def __init__(
        self,
        key: jnp.ndarray,
        world_radius: float,
        view_radius: float,
        map_size: int = 256,
        octaves: int = 4
    ):
        self.world_radius = world_radius
        # Generate heightmap once
        key, hm = generate_fractal_noise_2d(
            key,
            (map_size, map_size),
            res=(8, 8),
            octaves=octaves
        )
        self.heightmap = hm

    def __call__(self, xy: jnp.ndarray) -> jnp.ndarray:
        """
        Query terrain height for XY coordinates.
        """
        return interpolate_heightmap(
            self.heightmap,
            xy,
            (2 * self.world_radius, 2 * self.world_radius)
        )

# --- Spatial Partitioning Utilities ---
@jax.jit
def build_cell_list(pos_xy: jnp.ndarray, params: Params):
    """
    Compute spatial hash: assign each agent to a grid cell,
    return sorting order, prefix-sum starts, and cell IDs.
    """
    cell = jnp.floor((pos_xy + params.world_radius) / params.view_radius).astype(int)
    cell = jnp.clip(cell, 0, params.num_cells - 1)
    cid  = cell[:, 0] * params.num_cells + cell[:, 1]
    order   = jnp.argsort(cid)
    sorted_c = cid[order]
    counts  = jnp.bincount(sorted_c, length=params.num_cells_sq)
    starts  = jnp.concatenate([jnp.array([0]), jnp.cumsum(counts)])
    return order, starts, cid

@jax.jit
def gather_neighbors(
    order: jnp.ndarray,
    starts: jnp.ndarray,
    cid: jnp.ndarray,
    params: Params
) -> jnp.ndarray:
    """
    Fully JAX vectorized neighbor gather:
    For each agent, select K nearest agents in adjacent grid cells.
    No Python loops: uses boolean mask & argsort.
    """
    N = params.num_drones
    C = params.num_cells
    K = params.num_ent * 3
    # Compute 2D cell coordinates
    cell_xy = jnp.stack([cid // C, cid % C], axis=-1)
    # Pairwise cell differences
    ci = cell_xy[:, None, :]  # shape (N,1,2)
    cj = cell_xy[None, :, :]  # shape (1,N,2)
    dc = jnp.abs(ci - cj)     # shape (N,N,2)
    nearby = (dc[:, :, 0] <= 1) & (dc[:, :, 1] <= 1)  # shape (N,N)
    # For each agent, argsort mask to get first K neighbor indices
    def select_idxs(mask_row):
        idxs = jnp.argsort(~mask_row)[:K]
        return idxs
    return jax.vmap(select_idxs)(nearby)  # shape (N,K)

# Precompute ground sampling offsets
_lin = jnp.linspace; _mesh = jnp.meshgrid
GRID_OFFSETS = _mesh(
    _lin(-1.0, 1.0, Params.ground_res) * Params.view_radius,
    _lin(-1.0, 1.0, Params.ground_res) * Params.view_radius,
    indexing='xy'
)
GRID_OFFSETS = jnp.stack([GRID_OFFSETS[0].ravel(), GRID_OFFSETS[1].ravel()], -1)

# --- Main Environment API ---
@jax.jit
def reset(key: jnp.ndarray, params: Params):
    """
    Reset the environment: generate terrain and initial drone states.
    Returns: observations, state tuple.
    """
    # Split RNG for terrain, positions, velocities, teams, POIs
    key, k1, k2, k3, k4 = jax.random.split(key, 5)
    # Create Perlin terrain
    perlin_fn = Perlin2D(k1, params.world_radius, params.view_radius,
                         params.map_size, params.perlin_octaves)
    # Sample initial positions & velocities
    pos = jax.random.uniform(k2, (params.num_drones, 3),
                             -params.world_radius, params.world_radius)
    vel = jax.random.uniform(k3, (params.num_drones, 3),
                             params.min_speed, params.max_speed)
    vnorm = jnp.linalg.norm(vel, axis=1, keepdims=True)
    vel = vel * (jnp.clip(vnorm, params.min_speed, params.max_speed) / (vnorm + 1e-8))
    # Assign teams
    teams = jax.random.randint(k4, (params.num_drones,), 0, params.num_teams)
    # Sample POIs on terrain
    poi_xy = jax.random.uniform(k4, (params.num_poi, 2),
                                -params.world_radius, params.world_radius)
    poi_z  = perlin_fn(poi_xy)
    poi    = jnp.concatenate([poi_xy, poi_z[:, None]], axis=-1)
    # All drones start alive
    alive = jnp.ones(params.num_drones, dtype=bool)
    # Initialize state tuple
    state = (pos, vel, alive, teams, poi,
             perlin_fn.heightmap, 0, key)
    # Build spatial hash structures
    order, starts, cid = build_cell_list(pos[:, :2], params)
    neigh = gather_neighbors(order, starts, cid, params)
    state += (order, starts, cid, neigh)
    # Return initial obs and state
    obs = get_obs(state, params)
    return obs, state

@jax.jit
def step(state, action: jnp.ndarray, params: Params):
    """
    Advance one timestep: dynamics, collisions, and new observations.
    """
    # Unpack state
    (pos, vel, alive, teams, poi,
     hm, step_count, key,
     order, starts, cid, neigh) = state
    # 1) Integrate motion
    vel = vel + action * params.max_acceleration
    vnorm = jnp.linalg.norm(vel, axis=1, keepdims=True)
    vel = vel * (jnp.clip(vnorm, params.min_speed, params.max_speed) / (vnorm + 1e-8))
    pos = pos + vel
    # 2) Out-of-bounds check
    oob  = jnp.sum(pos[:, :2]**2, axis=1) > params.world_radius_sq
    # 3) Terrain collision
    land = pos[:, 2] < interpolate_heightmap(
        hm, pos[:, :2], (2*params.world_radius, 2*params.world_radius)
    )
    # 4) Rebuild spatial hash after moving agents
    order, starts, cid = build_cell_list(pos[:, :2], params)
    neigh = gather_neighbors(order, starts, cid, params)
    # 5) Drone-drone collisions via neighbor indices
    def coll(i):
        rels = pos[i] - pos[neigh[i]]
        return jnp.any(jnp.sum(rels**2, axis=1) < params.collision_radius_sq)
    collide = jax.vmap(coll)(jnp.arange(params.num_drones))
    # 6) Update alive flags and check done
    alive = alive & ~oob & ~land & ~collide
    done  = (step_count + 1 >= params.max_steps) | (~jnp.any(alive))
    # 7) Pack new state including spatial arrays
    new_state = (pos, vel, alive, teams, poi,
                 hm, step_count+1, key,
                 order, starts, cid, neigh)
    # 8) Compute observations
    obs = get_obs(new_state, params)
    return obs, new_state, jnp.zeros(params.num_drones), done, {}

@jax.jit
def get_obs(state, params: Params) -> jnp.ndarray:
    """
    Build per-agent observation vectors.
    """
    (pos, vel, alive, teams, poi,
     hm, step_count, key,
     order, starts, cid, neigh) = state
    N, E, v2 = params.num_drones, params.num_ent, params.view_radius_sq

    def one_obs(i):
        # Gather relative vectors to neighbor indices
        idxs = neigh[i]
        rels = pos[i] - pos[idxs]          # shape (K,3)
        d2   = jnp.sum(rels**2, axis=1)
        # Friendly neighbors
        mask_f = (teams[idxs] == teams[i]) & (d2 <= v2)
        idx_f  = jnp.argsort(jnp.where(mask_f, d2, jnp.inf))[:E]
        rf     = rels[idx_f]
        # Enemy neighbors
        mask_e = (teams[idxs] != teams[i]) & (d2 <= v2)
        idx_e  = jnp.argsort(jnp.where(mask_e, d2, jnp.inf))[:E]
        re     = rels[idx_e]
        # POI neighbors (range-based)
        rpoi   = poi - pos[i]              # shape (num_poi,3)
        d2p    = jnp.sum(rpoi**2, axis=1)
        mask_p = d2p <= v2
        idx_p  = jnp.argsort(jnp.where(mask_p, d2p, jnp.inf))[:E]
        rp     = rpoi[idx_p]
        # Distance from center ratio
        cr     = jnp.linalg.norm(pos[i, :2]) / params.world_radius
        # Ground patch sampling with circular mask
        grid_xy = pos[i, :2] + GRID_OFFSETS     # shape (G^2,2)
        d2g     = jnp.sum((grid_xy - pos[i, :2])**2, axis=1)
        hg      = interpolate_heightmap(
            hm, grid_xy, (2*params.world_radius, 2*params.world_radius)
        )
        go      = jnp.where(d2g <= v2, hg, 0.0)  # masked heights
        # Concatenate all features
        return jnp.concatenate([
            rf.ravel(), re.ravel(), rp.ravel(),
            jnp.array([cr]), go
        ], axis=0)

    return jax.vmap(one_obs)(jnp.arange(N))

@jax.jit
def get_obs_graph(state, params: Params) -> jnp.ndarray:
    """
    Build fixed-size graph nodes [dx,dy,dz,type_id] per agent.
    """
    (pos, vel, alive, teams, poi,
     hm, step_count, key,
     order, starts, cid, neigh) = state
    N, E, v2 = params.num_drones, params.num_ent, params.view_radius_sq

    def one_graph(i):
        idxs = neigh[i]
        rels = pos[i] - pos[idxs]
        d2   = jnp.sum(rels**2, axis=1)
        # Friendly nodes
        mask_f = (teams[idxs] == teams[i]) & (d2 <= v2)
        idx_f  = jnp.argsort(jnp.where(mask_f, d2, jnp.inf))[:E]
        nf     = rels[idx_f]
        # Enemy nodes
        mask_e = (teams[idxs] != teams[i]) & (d2 <= v2)
        idx_e  = jnp.argsort(jnp.where(mask_e, d2, jnp.inf))[:E]
        ne     = rels[idx_e]
        # POI nodes
        rpoi   = poi - pos[i]
        d2p    = jnp.sum(rpoi**2, axis=1)
        mask_p = d2p <= v2
        idx_p  = jnp.argsort(jnp.where(mask_p, d2p, jnp.inf))[:E]
        np_    = rpoi[idx_p]
        # Assign type IDs and concatenate
        tf = jnp.zeros((E,1)); te = jnp.ones((E,1)); tp = jnp.full((E,1),2.0)
        nf = jnp.concatenate([nf, tf], axis=1)
        ne = jnp.concatenate([ne, te], axis=1)
        np = jnp.concatenate([np_, tp], axis=1)
        return jnp.concatenate([nf, ne, np], axis=0)

    return jax.vmap(one_graph)(jnp.arange(N))

# Precompute ground sampling offsets (GRID_OFFSETS)
_lin = jnp.linspace; _mesh = jnp.meshgrid
GRID_OFFSETS = _mesh(
    _lin(-1.0, 1.0, Params.ground_res) * Params.view_radius,
    _lin(-1.0, 1.0, Params.ground_res) * Params.view_radius,
    indexing='xy'
)
GRID_OFFSETS = jnp.stack([GRID_OFFSETS[0].ravel(), GRID_OFFSETS[1].ravel()], -1)

# --- Integration with JaxMARL & Mava ---
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Box as JMBox

class SimpleTagEnv(MultiAgentEnv):
    """
    JaxMARL-compatible wrapper for dict-based observations/actions.
    """
    def __init__(self, params: Params):
        self.params = params
        self.agents = [f"agent_{i}" for i in range(params.num_drones)]
        # Observation space per agent
        obs_dim = params.num_ent*3*3 + 1 + params.ground_res**2
        self.observation_spaces = {
            a: JMBox(-jnp.inf, jnp.inf, (obs_dim,), jnp.float32)
            for a in self.agents
        }
        # Continuous 3D acceleration action space
        low, high = -params.max_acceleration, params.max_acceleration
        self.action_spaces = {
            a: JMBox(low, high, (3,), jnp.float32)
            for a in self.agents
        }

    def reset(self, key: jnp.ndarray):
        obs, state = reset(key, self.params)
        obs_dict = {a: obs[i] for i, a in enumerate(self.agents)}
        return obs_dict, state

    def step(self, key: jnp.ndarray, state: any, actions: dict):
        act_arr = jnp.stack([actions[a] for a in self.agents])
        obs, new_state, rews, done, info = step(state, act_arr, self.params)
        obs_dict = {a: obs[i] for i, a in enumerate(self.agents)}
        rew_dict = {a: float(rews[i]) for i, a in enumerate(self.agents)}
        done_dict= {a: done for a in self.agents}
        return obs_dict, new_state, rew_dict, done_dict, info

from jaxmarl.py import JaxMarlWrapper  # noqa: E402

def make_mava_env(params: Params):
    """
    Wrap SimpleTagEnv for Mava via JaxMarlWrapper.
    """
    env = SimpleTagEnv(params)
    return JaxMarlWrapper(env, has_global_state=False, time_limit=params.max_steps)

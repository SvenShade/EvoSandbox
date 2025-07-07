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
    shape: tuple[int, int],
    res: tuple[int, int],
    tileable: tuple[bool, bool] = (False, False),
    interpolant=interpolant
) -> jnp.ndarray:
    """
    Generate a single-octave Perlin noise heightmap of dimensions `shape`.
    """
    grid = jnp.stack(jnp.meshgrid(
        jnp.linspace(0, res[0], shape[0], endpoint=False) % 1,
        jnp.linspace(0, res[1], shape[1], endpoint=False) % 1,
        indexing='ij'
    ), axis=-1)
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    if tileable[0]: gradients[-1, :] = gradients[0, :]
    if tileable[1]: gradients[:, -1] = gradients[:, 0]
    d = (shape[0] // res[0], shape[1] // res[1])
    gradients = np.repeat(np.repeat(gradients, d[0], axis=0), d[1], axis=1)
    g00 = jnp.array(gradients[:-d[0], :-d[1]])
    g10 = jnp.array(gradients[d[0]:, :-d[1]])
    g01 = jnp.array(gradients[:-d[0], d[1]:])
    g11 = jnp.array(gradients[d[0]:, d[1]:])
    n00 = jnp.sum(grid * g00, axis=-1)
    n10 = jnp.sum((grid - jnp.array([1, 0])) * g10, axis=-1)
    n01 = jnp.sum((grid - jnp.array([0, 1])) * g01, axis=-1)
    n11 = jnp.sum((grid - jnp.array([1, 1])) * g11, axis=-1)
    t = interpolant(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return jnp.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def generate_fractal_noise_2d(
    shape: tuple[int,int],
    res: tuple[int,int],
    octaves: int = 4,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    tileable: tuple[bool,bool] = (False, False),
    interpolant=interpolant
) -> jnp.ndarray:
    """
    Generate fractal Perlin noise by summing multiple octaves.
    """
    noise = jnp.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(
            shape,
            (int(res[0] * frequency), int(res[1] * frequency)),
            tileable,
            interpolant
        )
        frequency *= lacunarity
        amplitude *= persistence
    return noise


def interpolate_heightmap(
    heightmap: jnp.ndarray,
    xy: jnp.ndarray,
    world_size: tuple[float, float]
) -> jnp.ndarray:
    """
    Bilinear interpolation of a heightmap at points xy.
    """
    H, W = heightmap.shape
    wx, wy = world_size
    grid_x = (xy[:, 0] + wx/2) * (W - 1) / wx
    grid_y = (xy[:, 1] + wy/2) * (H - 1) / wy
    x0 = jnp.floor(grid_x).astype(jnp.int32)
    x1 = jnp.clip(x0 + 1, 0, W - 1)
    y0 = jnp.floor(grid_y).astype(jnp.int32)
    y1 = jnp.clip(y0 + 1, 0, H - 1)
    wx_f = grid_x - x0
    wy_f = grid_y - y0
    h00 = heightmap[y0, x0]
    h10 = heightmap[y0, x1]
    h01 = heightmap[y1, x0]
    h11 = heightmap[y1, x1]
    return (
        h00 * (1 - wx_f) * (1 - wy_f) +
        h10 * wx_f       * (1 - wy_f) +
        h01 * (1 - wx_f) * wy_f       +
        h11 * wx_f       * wy_f
    )


class Perlin2D:
    """
    Precomputes a fractal Perlin noise map and interpolates on query.
    """
    def __init__(
        self,
        world_radius: float,
        view_radius: float,
        map_size: int = 256,
        octaves: int = 4
    ):
        self.world_radius = world_radius
        self.heightmap = generate_fractal_noise_2d((map_size, map_size), res=(8,8), octaves=octaves)

    def __call__(self, xy: jnp.ndarray) -> jnp.ndarray:
        return interpolate_heightmap(
            self.heightmap,
            xy,
            (2 * self.world_radius, 2 * self.world_radius)
        )

# --- Environment Definition ---

@dataclass
class Params:
    # Core environment parameters
    num_drones: int = 10             # number of drone agents
    num_teams: int = 2               # number of teams (friendly vs enemy)
    world_radius: float = 1.0        # radius of map
    max_acceleration: float = 0.01   # max acceleration per axis
    min_speed: float = 0.0           # minimum speed clamp
    max_speed: float = 0.03          # maximum speed clamp
    collision_radius: float = 0.05   # collision threshold for drones
    view_radius: float = 0.2         # visibility radius
    num_poi: int = 5                 # number of points of interest
    ground_res: int = 16             # resolution of ground patch grid
    num_ent: int = 5                 # max visible entities per type
    max_steps: int = 200             # max episode length
    # Precomputed squared thresholds for efficiency
    world_radius_sq: float = field(init=False)
    view_radius_sq: float = field(init=False)
    collision_radius_sq: float = field(init=False)

    def __post_init__(self):
        # Compute squared radii once
        self.world_radius_sq     = self.world_radius ** 2
        self.view_radius_sq      = self.view_radius ** 2
        self.collision_radius_sq = self.collision_radius ** 2

# Precompute ground patch offsets (square grid over [-view_radius,view_radius])
_lin = jnp.linspace; _mesh = jnp.meshgrid
GRID_OFFSETS = _mesh(
    _lin(-1.0, 1.0, Params.ground_res) * Params.view_radius,
    _lin(-1.0, 1.0, Params.ground_res) * Params.view_radius,
    indexing='xy'
)
GRID_OFFSETS = jnp.stack([GRID_OFFSETS[0].ravel(), GRID_OFFSETS[1].ravel()], -1)

@jax.jit
def reset(key: jnp.ndarray, params: Params):
    """
    Reset environment state and return initial observations.
    """
    key, subkey = jax.random.split(key)
    perlin_fn = Perlin2D(params.world_radius, params.view_radius)

    # Sample positions and velocities
    pk, vk, tk, xk = jax.random.split(subkey, 4)
    pos = jax.random.uniform(pk, (params.num_drones, 3), -params.world_radius, params.world_radius)
    vel = jax.random.uniform(vk, (params.num_drones, 3), params.min_speed, params.max_speed)
    vnorm = jnp.linalg.norm(vel, axis=1, keepdims=True)
    vel = vel * (jnp.clip(vnorm, params.min_speed, params.max_speed) / (vnorm + 1e-8))

    # Assign teams and PoIs
    teams = jax.random.randint(tk, (params.num_drones,), 0, params.num_teams)
    poi_xy = jax.random.uniform(xk, (params.num_poi, 2), -params.world_radius, params.world_radius)
    poi = jnp.concatenate([poi_xy, perlin_fn(poi_xy)[:, None]], axis=-1)

    alive = jnp.ones(params.num_drones, dtype=bool)
    state = (pos, vel, alive, teams, poi, perlin_fn, 0, key)
    obs, _ = get_obs(state, params), state
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

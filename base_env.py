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
    # Compute grid coordinates in [0, res) for interpolation
    grid = jnp.stack(jnp.meshgrid(
        jnp.linspace(0, res[0], shape[0], endpoint=False) % 1,
        jnp.linspace(0, res[1], shape[1], endpoint=False) % 1,
        indexing='ij'
    ), axis=-1)

    # Random gradient vectors at grid corners
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    if tileable[0]: gradients[-1, :] = gradients[0, :]
    if tileable[1]: gradients[:, -1] = gradients[:, 0]

    # Expand gradients over the shape
    d = (shape[0] // res[0], shape[1] // res[1])
    gradients = np.repeat(np.repeat(gradients, d[0], axis=0), d[1], axis=1)

    # Extract four corner gradients
    g00 = jnp.array(gradients[:-d[0], :-d[1]])
    g10 = jnp.array(gradients[d[0]:, :-d[1]])
    g01 = jnp.array(gradients[:-d[0], d[1]:])
    g11 = jnp.array(gradients[d[0]:, d[1]:])

    # Compute dot products (ramps)
    n00 = jnp.sum(grid * g00, axis=-1)
    n10 = jnp.sum((grid - jnp.array([1, 0])) * g10, axis=-1)
    n01 = jnp.sum((grid - jnp.array([0, 1])) * g01, axis=-1)
    n11 = jnp.sum((grid - jnp.array([1, 1])) * g11, axis=-1)

    # Interpolate
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
    Generate fractal (multi-octave) Perlin noise by summing multiple frequencies.
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
    Bilinearly interpolate the precomputed heightmap at points xy.
    """
    H, W = heightmap.shape
    wx, wy = world_size
    # Map xy in [-wx/2, wx/2] -> [0, W-1]
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
    Precomputes a fractal Perlin noise map and provides callable interpolation.
    """
    def __init__(
        self,
        world_radius: float,
        view_radius: float,
        map_size: int = 256,
        octaves: int = 4
    ):
        self.world_radius = world_radius
        self.map_size = map_size
        # build heightmap once
        self.heightmap = generate_fractal_noise_2d(
            (map_size, map_size),
            res=(8, 8),
            octaves=octaves
        )

    def __call__(self, xy: jnp.ndarray) -> jnp.ndarray:
        """
        Query terrain height at XY positions (shape: N×2).
        """
        return interpolate_heightmap(
            self.heightmap,
            xy,
            (2 * self.world_radius, 2 * self.world_radius)
        )


# --- Environment Definition ---

@dataclass
class Params:
    """
    Environment parameters, including squared thresholds for fast JIT.
    """
    num_drones: int = 10
    num_teams: int = 2
    world_radius: float = 1.0
    max_acceleration: float = 0.01
    min_speed: float = 0.0
    max_speed: float = 0.03
    collision_radius: float = 0.05
    view_radius: float = 0.2
    num_poi: int = 5
    ground_res: int = 16        # resolution of local ground grid
    num_ent: int = 5            # max entities per type
    max_steps: int = 200
    world_radius_sq: float = field(init=False)
    view_radius_sq: float = field(init=False)
    collision_radius_sq: float = field(init=False)

    def __post_init__(self):
        self.world_radius_sq     = self.world_radius**2
        self.view_radius_sq      = self.view_radius**2
        self.collision_radius_sq = self.collision_radius**2

# Precompute square grid offsets for ground patch sampling
lin = jnp.linspace
mesh = jnp.meshgrid
GRID_OFFSETS = _flat(*mesh(
    lin(-1.0, 1.0, Params.ground_res) * Params.view_radius,
    lin(-1.0, 1.0, Params.ground_res) * Params.view_radius,
    indexing='xy'
))  # shape: (ground_res**2, 2)


@jax.jit
def reset(key: jnp.ndarray, params: Params):
    """
    Initialize all agent states and terrain.
    Returns (observations, state).
    """
    key, subkey = jax.random.split(key)
    perlin_fn = Perlin2D(params.world_radius, params.view_radius)
    pk, vk, tk, xk = jax.random.split(subkey, 4)

    # random positions and velocities
    pos = jax.random.uniform(
        pk,
        (params.num_drones, 3),
        -params.world_radius,
        params.world_radius
    )
    vel = jax.random.uniform(
        vk,
        (params.num_drones, 3),
        params.min_speed,
        params.max_speed
    )
    vnorm = jnp.linalg.norm(vel, axis=1, keepdims=True)
    vel = vel * (jnp.clip(vnorm, params.min_speed, params.max_speed) / (vnorm + 1e-8))

    # assign teams and POIs
    teams = jax.random.randint(tk, (params.num_drones,), 0, params.num_teams)
    poi_xy = jax.random.uniform(
        xk,
        (params.num_poi, 2),
        -params.world_radius,
        params.world_radius
    )
    poi = jnp.concatenate([poi_xy, perlin_fn(poi_xy)[:, None]], axis=-1)

    alive = jnp.ones(params.num_drones, dtype=bool)
    state = (pos, vel, alive, teams, poi, perlin_fn, 0, key)
    return get_obs(state, params), state


@jax.jit
def step(state, action: jnp.ndarray, params: Params):
    """
    Advance environment by one timestep.
    Returns (obs, new_state, rewards, done, info).
    """
    pos, vel, alive, teams, poi, perlin_fn, step_count, key = state
    w2 = params.world_radius_sq
    c2 = params.collision_radius_sq

    # motion update
    vel = vel + action * params.max_acceleration
    vnorm = jnp.linalg.norm(vel, axis=1, keepdims=True)
    vel = vel * (jnp.clip(vnorm, params.min_speed, params.max_speed) / (vnorm + 1e-8))
    pos = pos + vel

    # out-of-bounds and terrain collision
    oob = jnp.sum(pos[:, :2]**2, axis=1) > w2
    land = pos[:, 2] < perlin_fn(pos[:, :2])

    # drone-drone collisions
    rel = pos[:, None, :] - pos[None, :, :]
    dist2 = jnp.sum(rel**2, axis=-1)
    collide = jnp.any(dist2 < c2, axis=1)

    alive = alive & ~oob & ~land & ~collide
    done = (step_count + 1 >= params.max_steps) | (~jnp.any(alive))

    new_state = (pos, vel, alive, teams, poi, perlin_fn, step_count + 1, key)
    return get_obs(new_state, params), new_state, jnp.zeros(params.num_drones), done, {}


@jax.jit
def get_obs(state, params: Params) -> jnp.ndarray:
    """
    Compute a concatenated observation vector for each drone.
    Includes neighbor rel-pos, POI rel-pos, center distance ratio, and ground patch.
    """
    pos, _, _, teams, poi, perlin_fn, _, _ = state
    N, E = params.num_drones, params.num_ent
    v2 = params.view_radius_sq

    # relative positions and visibility mask
    rel = pos[:, None, :] - pos[None, :, :]
    mask = 1 - jnp.eye(N, dtype=jnp.int32)
    dist2 = jnp.sum(rel**2, axis=-1)
    vis = dist2 <= v2

    # helper to select up to E visible entities
    def select(mask_mat, data):
        d2 = jnp.where(mask_mat, jnp.sum(data**2, axis=-1), jnp.inf)
        idx = jnp.argsort(d2, axis=1)[:, :E]
        return jnp.take_along_axis(data * mask_mat[:, :, None], idx[:, :, None], axis=1)

    # friendly & enemy neighbors
    ti, tj = teams[:, None], teams[None, :]
    rf = select((tj == ti) & vis, rel).reshape(N, -1)
    re = select((tj != ti) & vis, rel).reshape(N, -1)

    # POI neighbors
    rpoi = poi[None, :, :] - pos[:, None, :]
    rp = select(jnp.sum(rpoi**2, axis=-1) <= v2, rpoi).reshape(N, -1)

    # distance from center ratio
    cr = (jnp.linalg.norm(pos[:, :2], axis=1) / params.world_radius)[:, None]

    # ground patch sampling + circular mask
    global_xy = pos[:, None, :2] + GRID_OFFSETS[None, :, :]
    d2g = jnp.sum((global_xy - pos[:, None, :2])**2, axis=-1)
    hg = perlin_fn(global_xy.reshape(-1, 2)).reshape(N, -1)
    go = jnp.where(d2g <= v2, hg, 0.0)

    # concatenate all parts
    return jnp.concatenate([rf, re, rp, cr, go], axis=-1)


@jax.jit
def get_obs_graph(state, params: Params) -> jnp.ndarray:
    """
    Build a fixed-size graph node tensor per drone: (num_ent*3, 4).
    Node features: [dx, dy, dz, type_id] with type_id in {0,1,2}.
    """
    pos, _, _, teams, poi, perlin_fn, _, _ = state
    N, E = params.num_drones, params.num_ent
    v2 = params.view_radius_sq

    rel = pos[:, None, :] - pos[None, :, :]
    dist2 = jnp.sum(rel**2, axis=-1)
    vis = dist2 <= v2
    ti, tj = teams[:, None], teams[None, :]

    def _make_nodes(mask_mat, data, t_id: float) -> jnp.ndarray:
        d2 = jnp.where(mask_mat, jnp.sum(data**2, axis=-1), jnp.inf)
        idx = jnp.argsort(d2, axis=1)[:, :E]
        nodes = jnp.take_along_axis(data * mask_mat[:, :, None], idx[:, :, None], axis=1)
        types = jnp.full((N, E, 1), t_id)
        return jnp.concatenate([nodes, types], axis=-1)

    nf = _make_nodes((tj == ti) & vis, rel, 0.0)
    ne = _make_nodes((tj != ti) & vis, rel, 1.0)
    np_ = _make_nodes(jnp.sum((poi[None, :, :] - pos[:, None, :])**2, axis=-1) <= v2,
                       poi[None, :, :] - pos[:, None, :], 2.0)

    return jnp.concatenate([nf, ne, np_], axis=1)

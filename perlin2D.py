import math
from typing import Tuple

import jax
import jax.numpy as jnp

__all__ = ["Perlin2D"]

F32 = jnp.float32

@jax.jit
def _fade(t: jnp.ndarray) -> jnp.ndarray:
    """Perlin’s quintic interpolant 6t^5 – 15t^4 + 10t^3"""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

class Perlin2D:
    """Perlin‑noise generator.

    Parameters
    ----------
    shape : (int, int)
        Height × width of the noise map.
    res : (int, int)
        Number of periods along y and x. Shape must be divisible by res.
    key : jax.random.PRNGKey, optional
    """
    def __init__(self,
        shape: Tuple[int, int],
        res: Tuple[int, int],
        key: jax.Array | None = None,
    ) -> None:
        self.height, self.width = shape
        self.res_y, self.res_x = res
        if self.height % self.res_y or self.width % self.res_x:
            raise ValueError("`shape` must be an integer multiple of `res` along each axis.")
        self.key = key or jax.random.PRNGKey(0)
        self.dy = self.height // self.res_y
        self.dx = self.width // self.res_x

    def generate(
        self,
        *,
        octaves: int = 4,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
    ) -> jnp.ndarray:
        """Generate fractal/1‑f noise (sum of octaves)."""
        return _fractal_noise_2d(
            self.key,
            (self.height, self.width),
            (self.res_y, self.res_x),
            octaves,
            persistence,
            lacunarity,
        )

    @staticmethod
    @jax.jit
    def sample_height(
        heightmap: jnp.ndarray,
        xy: jnp.ndarray,
        world_extent: Tuple[float, float],
    ) -> jnp.ndarray:
        h, w = heightmap.shape
        wx, wy = world_extent

        ix = ((xy[:, 0] + wx) / (2 * wx)) * (w - 1)
        iy = ((xy[:, 1] + wy) / (2 * wy)) * (h - 1)

        x0 = jnp.clip(jnp.floor(ix).astype(jnp.int32), 0, w - 1)
        y0 = jnp.clip(jnp.floor(iy).astype(jnp.int32), 0, h - 1)
        x1 = jnp.clip(x0 + 1, 0, w - 1)
        y1 = jnp.clip(y0 + 1, 0, h - 1)

        sx = ix - x0.astype(F32)
        sy = iy - y0.astype(F32)

        h00 = heightmap[y0, x0]
        h10 = heightmap[y0, x1]
        h01 = heightmap[y1, x0]
        h11 = heightmap[y1, x1]

        i0 = h00 * (1 - sx) + h10 * sx
        i1 = h01 * (1 - sx) + h11 * sx
        return i0 * (1 - sy) + i1 * sy

@jax.jit
def _random_gradients(key: jax.Array, ry: int, rx: int) -> jnp.ndarray:
    """Unit‑length gradient vectors on a (ry+1)×(rx+1) lattice."""
    angles = jax.random.uniform(key, (ry + 1, rx + 1), minval=0.0, maxval=2 * math.pi)
    return jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)

@jax.jit
def _perlin_noise_2d(
    key: jax.Array,
    shape: Tuple[int, int],
    res: Tuple[int, int],
    cell_size: Tuple[int, int],
) -> jnp.ndarray:
    h, w = shape
    ry, rx = res
    dy, dx = cell_size

    # 1. Random gradient lattice
    g = _random_gradients(key, ry, rx)  # (ry+1, rx+1, 2)

    # 2. Fractional position within each cell
    grid_y = (jnp.arange(h) / dy).astype(F32)       # (H,)
    grid_x = (jnp.arange(w) / dx).astype(F32)       # (W,)
    yi = jnp.floor(grid_y).astype(jnp.int32)
    xi = jnp.floor(grid_x).astype(jnp.int32)
    yf = grid_y - yi
    xf = grid_x - xi

    yi = yi[:, None]
    yf = yf[:, None]
    xi = xi[None, :]
    xf = xf[None, :]

    # 3. Corner gradients
    g00 = g[yi, xi]
    g10 = g[yi, xi + 1]
    g01 = g[yi + 1, xi]
    g11 = g[yi + 1, xi + 1]

    # 4. Distance vectors & dot products
    d00 = jnp.stack([xf, yf], axis=-1)
    d10 = jnp.stack([xf - 1, yf], axis=-1)
    d01 = jnp.stack([xf, yf - 1], axis=-1)
    d11 = jnp.stack([xf - 1, yf - 1], axis=-1)

    s00 = jnp.sum(g00 * d00, axis=-1)
    s10 = jnp.sum(g10 * d10, axis=-1)
    s01 = jnp.sum(g01 * d01, axis=-1)
    s11 = jnp.sum(g11 * d11, axis=-1)

    # 5. Interpolation using fade curves
    u = _fade(xf)
    v = _fade(yf)

    nx0 = s00 * (1 - u) + s10 * u
    nx1 = s01 * (1 - u) + s11 * u
    nxy = nx0 * (1 - v) + nx1 * v

    # Normalise to roughly [‑1,1]
    return nxy * jnp.sqrt(2.0)

def _fractal_noise_2d(
    key: jax.Array,
    shape: Tuple[int, int],
    res: Tuple[int, int],
    octaves: int,
    persistence: float,
    lacunarity: float,
) -> jnp.ndarray:
    total = jnp.zeros(shape, dtype=F32)
    amp = 1.0
    freq_y, freq_x = 1, 1
    k = key
    for _ in range(octaves):
        k, subk = jax.random.split(k)
        total += amp * _perlin_noise_2d(
            subk,
            shape,
            (int(res[0] * freq_y), int(res[1] * freq_x)),
            (shape[0] // (res[0] * freq_y), shape[1] // (res[1] * freq_x)),
        )
        amp *= persistence
        freq_y *= lacunarity
        freq_x *= lacunarity

    # Scale so max possible amplitude ≈ 1
    normaliser = jnp.sum(jnp.array([persistence ** i for i in range(octaves)]))
    return total / normaliser

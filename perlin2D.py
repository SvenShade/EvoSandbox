import math
from typing import Tuple

import jax
import jax.numpy as jnp

__all__ = ["Perlin2D"]

F32 = jnp.float32

@jax.jit
def _fade(t: jnp.ndarray) -> jnp.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

@jax.jit
def _random_grads(key: jax.Array, ry: int, rx: int) -> jnp.ndarray:
    ang = jax.random.uniform(key, (ry + 1, rx + 1), minval=0.0, maxval=2 * math.pi)
    return jnp.stack([jnp.cos(ang), jnp.sin(ang)], axis=-1)  # (...,2)

@jax.jit
def _perlin2d(key: jax.Array, shape: Tuple[int, int], res: Tuple[int, int]) -> jnp.ndarray:
    h, w = shape
    ry, rx = res
    dy, dx = h // ry, w // rx

    g = _random_grads(key, ry, rx)  # (ry+1, rx+1, 2)

    # grid coords
    y, x = jnp.mgrid[:h, :w]
    y = y / dy
    x = x / dx
    yi = jnp.floor(y).astype(jnp.int32)
    xi = jnp.floor(x).astype(jnp.int32)
    yf = y - yi
    xf = x - xi

    # gather corner gradients
    g00 = g[yi, xi]
    g10 = g[yi, xi + 1]
    g01 = g[yi + 1, xi]
    g11 = g[yi + 1, xi + 1]

    # dot products
    def dot(gv, dx, dy):
        return jnp.sum(gv * jnp.stack([dx, dy], -1), -1)

    n00 = dot(g00, xf, yf)
    n10 = dot(g10, xf - 1, yf)
    n01 = dot(g01, xf, yf - 1)
    n11 = dot(g11, xf - 1, yf - 1)

    # interpolation
    u = _fade(xf)
    v = _fade(yf)
    nx0 = n00 * (1 - u) + n10 * u
    nx1 = n01 * (1 - u) + n11 * u
    nxy = nx0 * (1 - v) + nx1 * v

    return nxy * jnp.sqrt(2.0)  # [-1,1]


def _fractal2d(
    key: jax.Array,
    shape: Tuple[int, int],
    res: Tuple[int, int],
    octaves: int,
    persistence: float,
    lacunarity: float,
) -> jnp.ndarray:
    total = jnp.zeros(shape, F32)
    amp = 1.0
    fy, fx = 1, 1
    k = key
    for _ in range(octaves):
        k, subk = jax.random.split(k)
        total += amp * _perlin2d(subk, shape, (int(res[0] * fy), int(res[1] * fx)))
        amp *= persistence
        fy *= lacunarity
        fx *= lacunarity
    norm = jnp.sum(jnp.array([persistence ** i for i in range(octaves)], dtype=F32))
    return total / norm

class Perlin2D:
    """Generate a fractal (1/f) Perlin heightmap.
    """
    def __init__(
        self,
        shape: Tuple[int, int],
        res: Tuple[int, int],
        *,
        octaves: int = 4,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        key: jax.Array | None = None,
    ) -> None:
        if shape[0] % res[0] or shape[1] % res[1]:
            raise ValueError("`shape` must be divisible by `res`.")
        key = key or jax.random.PRNGKey(0)
        self.heightmap = _fractal2d(key, shape, res, octaves, persistence, lacunarity)
        self.h, self.w = shape

    @jax.jit
    def sample_height(self, xy: jnp.ndarray) -> jnp.ndarray:
        """Bilinear height lookup at normalised coords, [-1, 1].
        """
        ix = ((xy[:, 0] + 1.0) * 0.5) * (self.w - 1)
        iy = ((xy[:, 1] + 1.0) * 0.5) * (self.h - 1)

        x0 = jnp.clip(jnp.floor(ix).astype(jnp.int32), 0, self.w - 1)
        y0 = jnp.clip(jnp.floor(iy).astype(jnp.int32), 0, self.h - 1)
        x1 = jnp.clip(x0 + 1, 0, self.w - 1)
        y1 = jnp.clip(y0 + 1, 0, self.h - 1)

        sx = ix - x0.astype(F32)
        sy = iy - y0.astype(F32)

        h00 = self.heightmap[y0, x0]
        h10 = self.heightmap[y0, x1]
        h01 = self.heightmap[y1, x0]
        h11 = self.heightmap[y1, x1]

        i0 = h00 * (1 - sx) + h10 * sx
        i1 = h01 * (1 - sx) + h11 * sx
        return i0 * (1 - sy) + i1 * sy

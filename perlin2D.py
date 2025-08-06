"""Pure‑JAX Perlin‑noise helper (square‑map, minimal).

* Builds a **fractal** height‑map once during construction.
* Map is square: size × size, values scaled to **[0, 1]**.
* Only public call: `sample_height(xy)` for bilinear lookup at normalised
  coordinates **[0,1]²** (top‑left origin).
"""

from __future__ import annotations

import math
import jax
import jax.numpy as jnp

__all__ = ["Perlin2D"]
F32 = jnp.float32

# --------------------------------------------------------
#  Fade (smoothstep⁵) ------------------------------------
# --------------------------------------------------------

@jax.jit
def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

# --------------------------------------------------------
#  Single‑octave Perlin for square map -------------------
# --------------------------------------------------------

@jax.jit(static_argnames=("size", "res"))
def _perlin2d(key, *, size: int, res: int):
    """Return one octave of Perlin noise in **[‑1,1]** for a *size×size* grid."""
    cell = size // res  # pixels per lattice cell
    ang = jax.random.uniform(key, (res + 1, res + 1)) * (2 * math.pi)
    g = jnp.stack([jnp.cos(ang), jnp.sin(ang)], -1)  # (res+1,res+1,2)

    y, x = jnp.mgrid[:size, :size]
    y = y / cell
    x = x / cell
    yi = jnp.floor(y).astype(jnp.int32)
    xi = jnp.floor(x).astype(jnp.int32)
    yf = y - yi
    xf = x - xi

    g00 = g[yi, xi]
    g10 = g[yi, xi + 1]
    g01 = g[yi + 1, xi]
    g11 = g[yi + 1, xi + 1]

    def dot(gv, dx, dy):
        return jnp.sum(gv * jnp.stack([dx, dy], -1), -1)

    n00 = dot(g00, xf, yf)
    n10 = dot(g10, xf - 1, yf)
    n01 = dot(g01, xf, yf - 1)
    n11 = dot(g11, xf - 1, yf - 1)

    u = _fade(xf)
    v = _fade(yf)
    nx0 = n00 * (1 - u) + n10 * u
    nx1 = n01 * (1 - u) + n11 * u
    nxy = nx0 * (1 - v) + nx1 * v
    return nxy * jnp.sqrt(2.0)

# --------------------------------------------------------
#  Fractal sum -------------------------------------------
# --------------------------------------------------------

@jax.jit(static_argnames=("size", "res", "octaves", "persistence", "lacunarity"))
def _fractal2d(key, *, size: int, res: int, octaves: int, persistence: float, lacunarity: float):
    total = jnp.zeros((size, size), F32)
    amp = 1.0
    freq = 1
    k = key
    for _ in range(octaves):
        k, subk = jax.random.split(k)
        total += amp * _perlin2d(subk, size=size, res=int(res * freq))
        amp *= persistence
        freq *= lacunarity
    norm = jnp.sum(jnp.array([persistence ** i for i in range(octaves)], F32))
    return total / norm

# --------------------------------------------------------
#  Public wrapper ----------------------------------------
# --------------------------------------------------------

class Perlin2D:
    """Square fractal height‑map (values ∈ [0,1])."""

    def __init__(
        self,
        size: int,
        res: int,
        *,
        octaves: int = 4,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        key: jax.Array | None = None,
    ) -> None:
        if size % res:
            raise ValueError("`size` must be divisible by `res`.")
        key = key or jax.random.PRNGKey(0)
        raw = _fractal2d(key, size=size, res=res, octaves=octaves, persistence=persistence, lacunarity=lacunarity)
        self.heightmap = (raw + 1.0) * 0.5  # shift to [0,1]
        self.size = size

    # ----------------------------------------------
    #  Bilinear sampler
    # ----------------------------------------------

    @jax.jit
    def sample_height(self, xy):
        """Return height at normalised coords **[0,1]²** (shape (N,2))."""
        ix = xy[:, 0] * (self.size - 1)
        iy = xy[:, 1] * (self.size - 1)

        x0 = jnp.floor(ix).astype(jnp.int32)
        y0 = jnp.floor(iy).astype(jnp.int32)
        x1 = jnp.clip(x0 + 1, 0, self.size - 1)
        y1 = jnp.clip(y0 + 1, 0, self.size - 1)

        sx = ix - x0.astype(F32)
        sy = iy - y0.astype(F32)

        h00 = self.heightmap[y0, x0]
        h10 = self.heightmap[y0, x1]
        h01 = self.heightmap[y1, x0]
        h11 = self.heightmap[y1, x1]

        i0 = h00 * (1 - sx) + h10 * sx
        i1 = h01 * (1 - sx) + h11 * sx
        return i0 * (1 - sy) + i1 * sy

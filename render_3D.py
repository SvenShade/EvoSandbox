    def render(
        self,
        p_pos: jnp.ndarray,
        dones: jnp.ndarray,
        heightmap: jnp.ndarray,
        *,
        radius: float = 0.06,
        elev: float = 45,
        azim: float = 45,
        cmap: str = "terrain",
        show: bool = True,
    ):
        """Plot a 3‑D frame showing terrain surface + spherical agents."""
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused

        hm = np.asarray(heightmap)
        h, w = hm.shape
        X, Y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, hm, cmap=cmap, linewidth=0, antialiased=False, alpha=0.8)

        # Sphere template
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        su = np.outer(np.cos(u), np.sin(v))
        sv = np.outer(np.sin(u), np.sin(v))
        sw = np.outer(np.ones_like(u), np.cos(v))

        for pos, done in zip(np.asarray(p_pos), np.asarray(dones)):
            cx, cy, cz = pos
            xs = radius * su + cx / self.world_size
            ys = radius * sv + cy / self.world_size
            zs = radius * sw + cz / self.world_size
            color = "grey" if done else "tab:blue"
            ax.plot_surface(xs, ys, zs, color=color, shade=True, linewidth=0)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.tight_layout()
        if show:
            plt.show()
        return fig




def wall_from_col(col_idx):
    # stack boundary column with same X/Y but Z at floor
    Xw = np.vstack([X[:, col_idx], X[:, col_idx]])
    Yw = np.vstack([Y[:, col_idx], Y[:, col_idx]])
    Zw = np.vstack([Z[:, col_idx], np.full_like(Z[:, col_idx], zfloor)])
    ax.plot_surface(Xw, Yw, Zw, color='lightgrey', linewidth=0, antialiased=False)

def wall_from_row(row_idx):
    Xw = np.vstack([X[row_idx, :], X[row_idx, :]])
    Yw = np.vstack([Y[row_idx, :], Y[row_idx, :]])
    Zw = np.vstack([Z[row_idx, :], np.full_like(Z[row_idx, :], zfloor)])
    ax.plot_surface(Xw, Yw, Zw, color='lightgrey', linewidth=0, antialiased=False)

# Four side walls
wall_from_col(0)         # left
wall_from_col(-1)        # right
wall_from_row(0)         # front
wall_from_row(-1)        # back

# Optional bottom (the floor rectangle)
ax.plot_surface(X, Y, np.full_like(Z, zfloor), color='gainsboro', linewidth=0)


import subprocess; subprocess.run(["ffmpeg","-y","-i","input.avi","-c:v","libx264","-crf","23","-preset","veryfast","-pix_fmt","yuv420p","-movflags","+faststart","-an","output.mp4"], check=True)


@jax.jit
def level_heightmap_square(
    heightmap: chex.Array,      # shape (H, H)
    points: chex.Array,         # shape (N, 2), (x, y) pixel coords, N >= 2
    radius: Union[float, int],  # scalar radius
) -> chex.Array:
    """
    For each point, set all pixels inside the circle (centered at that point,
    with the given radius) to the height at the point’s indexed (rounded, clipped) pixel.
    Later points win in overlaps.

    Assumptions:
      - heightmap is square (H x H)
      - points has shape (N, 2) with N >= 2
    """
    hm = jnp.asarray(heightmap)
    H = hm.shape[-1]
    # Quick shape check (works under JIT because shapes are static)
    if hm.shape[-2] != H:
        raise ValueError("heightmap must be square (H x H).")

    pts = jnp.asarray(points, dtype=hm.dtype)
    if pts.ndim != 2 or pts.shape[-1] != 2:
        raise ValueError("`points` must have shape (N, 2) with N >= 2.")

    r = jnp.asarray(radius, dtype=hm.dtype)
    r2 = r * r

    def body(i, cur_hm):
        cx, cy = pts[i, 0], pts[i, 1]

        # Target height from this point's (rounded, clipped) pixel
        ix = jnp.clip(jnp.round(cx).astype(jnp.int32), 0, H - 1)
        iy = jnp.clip(jnp.round(cy).astype(jnp.int32), 0, H - 1)
        h_target = cur_hm[iy, ix].astype(cur_hm.dtype)

        # Bounded box around the circle (exclusive ends for slicing)
        x0 = jnp.maximum(0, jnp.floor(cx - r).astype(jnp.int32))
        y0 = jnp.maximum(0, jnp.floor(cy - r).astype(jnp.int32))
        x1 = jnp.minimum(H, jnp.ceil(cx + r).astype(jnp.int32) + 1)
        y1 = jnp.minimum(H, jnp.ceil(cy + r).astype(jnp.int32) + 1)

        sub = lax.dynamic_slice(cur_hm, (y0, x0), (y1 - y0, x1 - x0))

        # Local distance mask inside the box
        yy = jnp.arange(y0, y1, dtype=hm.dtype).reshape(-1, 1)
        xx = jnp.arange(x0, x1, dtype=hm.dtype).reshape(1, -1)
        dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        mask = dist2 <= r2

        new_sub = jnp.where(mask, h_target, sub)
        return lax.dynamic_update_slice(cur_hm, new_sub, (y0, x0))

    return lax.fori_loop(0, pts.shape[0], body, hm)

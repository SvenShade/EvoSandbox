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


def _nearest_sample(img, y, x):
    H, W = img.shape
    yi = jnp.clip(jnp.round(y).astype(jnp.int32), 0, H - 1)
    xi = jnp.clip(jnp.round(x).astype(jnp.int32), 0, W - 1)
    return img[yi, xi]

def aerials(
    heightmap: jnp.ndarray,      # [S, S] float
    terrain_mask: jnp.ndarray,   # [S, S] bool
    pos: jnp.ndarray,            # [N, 3] (x, y, z)
    *,
    view_radius: float,          # spherical radius
    distance_scale: float = 50.0
):
    N = pos.shape[0]
    half = (2 * ceil(view_radius) + 1) // 2

    offs = jnp.arange(-half, half + 1, dtype=jnp.float32)
    dy = offs[None, :, None]                      # [1, S, 1]
    dx = offs[None, None, :]                      # [1, 1, S]

    x = pos[:, 0][:, None, None]
    y = pos[:, 1][:, None, None]
    z = pos[:, 2][:, None, None]

    X = x + dx
    Y = y + dy

    # Nearest-neighbor for both height and mask (mask is boolean -> float)
    H_patch = _nearest_sample(heightmap, Y, X)    # [N, S, S]
    M_patch = _nearest_sample(terrain_mask, Y, X).astype(jnp.float32)

    # Clearance and encoding
    clearance = jnp.maximum(z - H_patch, 0.0)
    mag = jnp.clip(clearance / distance_scale, 0.0, 1.0)
    sign = jnp.where(M_patch > 0.5, 1.0, -1.0)
    vals = sign * mag                               # [-1,1], 0 at contact

    # Spherical visibility
    dist3 = jnp.sqrt(dx**2 + dy**2 + clearance**2)
    inside = dist3 <= view_radius

    snapshots = jnp.where(inside, vals, -1.0)
    return snapshots

grid.point_data["mask"] = mask.astype(np.uint8).ravel(order="F")

# quick view (two colors), no scalar bar
grid.plot(
    scalars="mask",
    cmap=["#cccccc", "#ff6f61"],   # [False color, True color]
    show_scalar_bar=False,
    lighting=True,
)


def rear_facing_mask(pos, vel, max_dist, cos_thresh=0.8660254037844386):
    # pos, vel: (N,3).  cos_thresh = cos(half_angle); 0.866... ≈ cos(30°)
    eps = 1e-8
    vhat = vel / (jnp.linalg.norm(vel, axis=-1, keepdims=True) + eps)         # (N,1,3) later
    r = pos[None, :, :] - pos[:, None, :]                                      # r_ij: i -> j
    d = jnp.linalg.norm(r, axis=-1) + eps
    u = r / d[..., None]                                                       # unit i->j
    a = jnp.einsum('ik,ijk->ij', vhat, u)                                      # i faces j
    b = jnp.einsum('jk,ijk->ij', vhat, u)                                      # j faces away from i (rear of j)
    within = (d <= max_dist) & (~jnp.eye(pos.shape[0], dtype=bool))
    return within & (a >= cos_thresh) & (b >= cos_thresh)

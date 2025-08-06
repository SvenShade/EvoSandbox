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

def _explore_illumination_now(self, state: State) -> chex.Array:
    """
    Current-step illumination map M_t: binary disks (no falloff), summed over agents.
    Shape: (env_size, env_size).
    """
    S = int(self.env_size)
    eps = self.eps

    # Max kernel radius in cells (static size for stamping)
    R = int(jnp.ceil(self.view_rad))
    K = 2 * R + 1

    # Precompute distance patch centered at (R, R): shape (K, K)
    yy, xx = jnp.meshgrid(
        jnp.arange(K, dtype=F32),
        jnp.arange(K, dtype=F32),
        indexing="ij",
    )
    dy = yy - F32(R)
    dx = xx - F32(R)
    dist_patch = jnp.sqrt(dx * dx + dy * dy)  # (K, K)

    # Agent positions
    pos = state.p_pos.astype(F32)   # (N, 3)
    xy = pos[:, :2]
    z  = pos[:, 2]

    # Terrain height under agents
    norm_xy = jnp.clip(xy / F32(self.env_size), 0.0, 1.0)
    terrain_h = sample_hmap(state.hmap, norm_xy) * F32(self.env_size)  # (N,)
    alt = jnp.maximum(z - terrain_h, 0.0)

    # Radius shrinks linearly with altitude: full at ground -> 0 at alt >= view_rad
    rad_scale = jnp.clip(F32(1.0) - alt / (F32(self.view_rad) + eps), 0.0, 1.0)
    rad = F32(self.view_rad) * rad_scale  # (N,)

    # Integer centers (grid coords)
    cx = jnp.clip(jnp.round(xy[:, 0]).astype(jnp.int32), 0, S - 1)
    cy = jnp.clip(jnp.round(xy[:, 1]).astype(jnp.int32), 0, S - 1)

    # Padded canvas so every KxK stamp is in-bounds
    canvas = jnp.zeros((S + 2 * R, S + 2 * R), dtype=F32)

    def splat_one(i, canv):
        r = rad[i]

        # Linear falloff: 1 at center, 0 at radius r, 0 outside.
        # If r is tiny, this becomes ~0 everywhere; keep a dot at center in that case.
        kern = jnp.clip(F32(1.0) - dist_patch / (r + eps), 0.0, 1.0)
        kern = jnp.where(r < F32(0.75), (dist_patch == 0).astype(F32), kern)

        y0 = cy[i]
        x0 = cx[i]

        patch = lax.dynamic_slice(canv, (y0, x0), (K, K))
        patch = patch + kern
        return lax.dynamic_update_slice(canv, patch, (y0, x0))

    canvas = lax.fori_loop(0, pos.shape[0], splat_one, canvas)
    return canvas[R:R + S, R:R + S] 

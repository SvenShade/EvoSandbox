import jax
import jax.numpy as jnp
from jax import lax

def make_conv2d_same_stride1_c1_to_cout(
    H: int,
    W: int,
    C_out: int,
    kernel: jnp.ndarray,
    bias: jnp.ndarray | None = None,
):
    """
    x:      (B, N, H, W, 1)   bf16 recommended
    kernel: (K, K, 1, C_out)  (HWIO)
    bias:   (C_out,) optional
    y:      (B, N, H, W, C_out)

    Uses SAME padding, stride=1.
    """
    assert kernel.ndim == 4 and kernel.shape[2] == 1 and kernel.shape[3] == C_out
    K_h, K_w = kernel.shape[0], kernel.shape[1]

    # Precompute SAME padding tuples for (H, W)
    pad_h = (K_h - 1) // 2
    pad_w = (K_w - 1) // 2
    padding = [(pad_h, pad_h), (pad_w, pad_w)]  # SAME when K is odd (e.g. 3)

    # Dimension numbers for NHWC x HWIO -> NHWC
    dn = lax.conv_dimension_numbers(
        lhs_shape=(1, H, W, 1),
        rhs_shape=kernel.shape,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )

    # Close over fixed params for efficiency
    k = kernel
    b = bias

    @jax.jit
    def apply(x: jnp.ndarray) -> jnp.ndarray:
        assert x.shape[2:] == (H, W, 1)
        B, N = x.shape[0], x.shape[1]

        # Merge (B,N) into a single batch for conv: (B*N, H, W, 1)
        x2 = x.reshape(B * N, H, W, 1)

        y2 = lax.conv_general_dilated(
            lhs=x2,
            rhs=k,
            window_strides=(1, 1),
            padding=padding,
            dimension_numbers=dn,
            lhs_dilation=None,
            rhs_dilation=None,
            feature_group_count=1,  # regular conv (not depthwise/grouped)
        )  # (B*N, H, W, C_out)

        if b is not None:
            y2 = y2 + b  # broadcasts over (B*N, H, W)

        # Restore (B,N,...) shape
        return y2.reshape(B, N, H, W, C_out)

    return apply


B, N, H, W = 32, 4, 9, 9
C_out = 16
K = 3

key = jax.random.key(0)
x = jax.random.normal(key, (B, N, H, W, 1), dtype=jnp.bfloat16)

k_key, b_key = jax.random.split(key)
kernel = jax.random.normal(k_key, (K, K, 1, C_out), dtype=jnp.bfloat16)
bias = jnp.zeros((C_out,), dtype=jnp.bfloat16)

conv = make_conv2d_same_stride1_c1_to_cout(H, W, C_out, kernel, bias=bias)
y = conv(x)  # (B, N, H, W, C_out)


@@
     def get_obs(self, diff_pos, sqr_dist, aerials, state: State
         ) -> Dict[str, chex.Array]:       
         # Agent observations:
         #   + self pos, vel, fire, team ID, battery
         #   + diff pos, vel, fire, and team IDs of closest neighbours
-        #   + aerial snapshot of terrain
+        #   + terrain encoding (radial star sampling of snapshot, rotated by agent yaw)

+        # ---------------------------------------------------------------------
+        # Terrain star sampling (JIT-friendly; no Python lists)
+        # `exploration` snapshot already exists as state.snapshots: [N, W, W] in ~[0,1]
+        # We sample R rays x S samples along each ray across the patch, with bilinear
+        # interpolation, and rotate rays by each agent's yaw inferred from state.p_ori.
+        # Defaults keep terrain_dim unchanged: R=S=view_width -> R*S == view_width**2.
+        # ---------------------------------------------------------------------
+        num_rays    = int(getattr(self, "terrain_rays",    self.view_width))
+        num_samples = int(getattr(self, "terrain_samples", self.view_width))
+
+        W  = state.snapshots.shape[1]
+        cx = (F32(W) - F32(1.0)) * F32(0.5)  # patch center in pixel coords
+
+        # Base ray angles (in patch/global frame), then rotate by agent yaw.
+        base_theta = F32(2.0) * jnp.pi * (jnp.arange(num_rays, dtype=F32) / F32(num_rays))  # [R]
+        yaw = jnp.arctan2(state.p_ori[:, 1], state.p_ori[:, 0]).astype(F32)                 # [N]
+        theta = (yaw[:, None] + base_theta[None, :]).astype(F32)[:, :, None]                # [N,R,1]
+
+        # Samples along each ray from edge->center->edge (t in [-1, 1]).
+        t = jnp.linspace(F32(-1.0), F32(1.0), num_samples, dtype=F32)[None, None, :]        # [1,1,S]
+
+        # Ray coordinates in patch pixel space: (x,y) in [0, W-1]
+        x = cx + (t * jnp.cos(theta) * cx)   # [N,R,S]
+        y = cx + (t * jnp.sin(theta) * cx)   # [N,R,S]
+
+        def _bilinear_sample(img: chex.Array, x: chex.Array, y: chex.Array) -> chex.Array:
+            """img: [W,W], x/y: [R,S] float pixel coords -> out: [R,S]"""
+            eps = self.eps
+            img = img.astype(F32)
+
+            x = jnp.clip(x, F32(0.0), F32(W - 1) - eps)
+            y = jnp.clip(y, F32(0.0), F32(W - 1) - eps)
+
+            x0 = jnp.floor(x).astype(jnp.int32)
+            y0 = jnp.floor(y).astype(jnp.int32)
+            x1 = jnp.minimum(x0 + 1, W - 1)
+            y1 = jnp.minimum(y0 + 1, W - 1)
+
+            wx = (x - x0.astype(F32))
+            wy = (y - y0.astype(F32))
+
+            Ia = img[y0, x0]
+            Ib = img[y0, x1]
+            Ic = img[y1, x0]
+            Id = img[y1, x1]
+
+            wa = (F32(1.0) - wx) * (F32(1.0) - wy)
+            wb = wx * (F32(1.0) - wy)
+            wc = (F32(1.0) - wx) * wy
+            wd = wx * wy
+
+            return wa * Ia + wb * Ib + wc * Ic + wd * Id
+
+        # Vectorise bilinear sampling across agents: -> star: [N,R,S]
+        star = jax.vmap(_bilinear_sample, in_axes=(0, 0, 0))(state.snapshots, x, y).astype(F32)
+        star_flat = star.reshape((self.num_agnt, num_rays * num_samples))  # [N, R*S]
+
         def _obs(i: int):
             # Find indices of closest neighbours, nearest first.
             dists = jnp.where(sqr_dist[i] > 0.0, sqr_dist[i], jnp.inf)
             nbor_idxs = jnp.argsort(dists, axis=-1)[:self.obs_slots]
@@
             # Concatenate self, neighbour, and terrain obs.
             return jnp.concatenate(
                 [self_pos, self_vel, self_att, self_fre, self_bat, self_act,
                  nbor_pos, nbor_vel, nbor_att, nbor_fre, nbor_tem,
-                 state.snapshots[i].flatten() * 2 - 1]
+                 star_flat[i] * 2 - 1]
             )
         
         return {a: _obs(i) for i, a in enumerate(self.agents)}

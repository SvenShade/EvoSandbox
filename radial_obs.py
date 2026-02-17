from typing import Callable, Sequence
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import lax


import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn


class FastConv9x9_C1_K3S1(nn.Module):
    """Exact 3x3 SAME conv specialized for (...,9,9,1) -> (...,9,9,C_out)."""
    features: int
    use_bias: bool = True
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.float32
    kernel_init: callable = nn.initializers.lecun_normal()
    bias_init: callable = nn.initializers.zeros
    pad_to: int | None = 16  # pad K=9 to 16 for bf16 matmul kernels (benchmark!)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x)
        if x.shape[-3:] != (9, 9, 1):
            raise ValueError(f"Expected (...,9,9,1); got {x.shape}")

        lead = x.shape[:-3]
        if lead:
            Bflat = int(np.prod(lead))
            x2 = x.reshape(Bflat, 9, 9, 1)
        else:
            x2 = x  # (B,9,9,1)

        x2 = x2.astype(self.dtype)

        # Pad once: (B, 11, 11, 1)
        xp = jnp.pad(x2, ((0,0), (1,1), (1,1), (0,0)))

        # Collect the 9 shifted views (each is (B,9,9,1)), then stack -> (B,9,9,9)
        shifts = []
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                shifts.append(xp[:, dy:dy+9, dx:dx+9, 0])  # (B,9,9)
        X9 = jnp.stack(shifts, axis=-1)  # (Bflat, 9, 9, 9)

        # Learned weights: (9, C_out)
        W = self.param("kernel", self.kernel_init, (9, self.features)).astype(self.param_dtype)
        W = W.astype(self.dtype)

        # Optional pad to 16 to help bf16 matmul
        if self.pad_to is not None:
            Kpad = int(self.pad_to * ((9 + self.pad_to - 1) // self.pad_to))
            if Kpad != 9:
                X9 = jnp.pad(X9, ((0,0),(0,0),(0,0),(0, Kpad-9)))
                W = jnp.pad(W, ((0, Kpad-9), (0,0)))

        # One GEMM over last dim: (B*81, K) @ (K, C_out)
        Bflat = X9.shape[0]
        K = X9.shape[-1]
        Y = (X9.reshape(Bflat * 81, K) @ W).reshape(Bflat, 9, 9, self.features)

        if self.use_bias:
            b = self.param("bias", self.bias_init, (self.features,)).astype(self.param_dtype)
            Y = Y + b.astype(self.dtype)

        if lead:
            return Y.reshape(*lead, 9, 9, self.features)
        return Y




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

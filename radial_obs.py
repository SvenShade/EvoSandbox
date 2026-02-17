from typing import Callable, Sequence
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import lax


# Terrain encoder: patchify 3x3 (stride 1) -> token MLP-mixer (matmul-friendly)
# Input `ter` is (B, N, view, view, 1) with view=9 by default.
BN = B * N
tview = view
# 3x3 valid patches (stride 1, VALID padding): (BN, Hout, Wout, 9) for C_in=1.
# Use XLA's patch extractor (im2col) so this works for any `view` without manual slicing.
b = ter.reshape(BN, tview, tview, 1).astype(jnp.bfloat16)
patches = jax.lax.conv_general_dilated_patches(
    b,
    filter_shape=(3, 3),
    window_strides=(1, 1),
    padding="VALID",
    dimension_numbers=("NHWC", "HWIO", "NHWC"),
)  # (BN, Hout, Wout, 9)

Hout, Wout = patches.shape[1], patches.shape[2]
T = Hout * Wout  # tokens = Hout*Wout
tok = patches.reshape(BN, T, patches.shape[-1])

# Per-token embedding (GEMM-friendly).
C = 16
tok = nn.Dense(C, kernel_init=orthogonal(np.sqrt(2)))(tok)
tok = self.act(tok)

# Token-mixing MLP (Mixer-style): mixes across the token dimension with dense layers.
# This keeps the locality signal from patchification but avoids small convolutions.
tm = 64  # token-mixing hidden
y = jnp.swapaxes(tok, 1, 2)  # (BN, C, T)
y = nn.Dense(tm, kernel_init=orthogonal(np.sqrt(2)))(y)
y = self.act(y)
y = nn.Dense(T, kernel_init=orthogonal(np.sqrt(2)))(y)
y = jnp.swapaxes(y, 1, 2)  # (BN, T, C)
tok = tok + y

# Pool tokens to a fixed-size vector per agent.
tok_mx = jnp.max(tok, axis=1)
tok_mn = jnp.mean(tok, axis=1)
ter_vec = jnp.concatenate([tok_mx, tok_mn], axis=-1)  # (BN, 2C)

ter_out = 64
ter_vec = nn.Dense(ter_out, kernel_init=orthogonal(np.sqrt(2)))(ter_vec)
ter_vec = self.act(ter_vec)

ter = ter_vec.reshape(B, N, ter_out)



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

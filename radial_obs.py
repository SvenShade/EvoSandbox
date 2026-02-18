from typing import Callable, Sequence
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import lax

import flax.linen as nn
import jax.numpy as jnp
import numpy as np


self.obs_slots    = 5  # Fixed neighbor slots (padding when fewer in range)
# Precompute upper-tri indices for pairwise neighbour geometry (K choose 2).
_pi, _pj = np.triu_indices(self.obs_slots, k=1)
self.pair_i = jnp.array(_pi, dtype=jnp.int32)
self.pair_j = jnp.array(_pj, dtype=jnp.int32)


# Invariant neighbour scalars (rotation-invariant features).
# These are expressed in the same normalised frame as nbor_pos/nbor_vel.
#   nbor_pos is already / view_rad   -> components in [-1, 1]
#   nbor_vel is already / max_speed -> components in (roughly) [-1, 1]
# We scale norms by sqrt(3) so that max norm of a [-1,1]^3 vector maps to ~1.
sqrt3 = jnp.sqrt(PRE(3.0))
pos_norm = jnp.linalg.norm(nbor_pos, axis=-1)  # (K,)
vel_norm = jnp.linalg.norm(nbor_vel, axis=-1)  # (K,)

# range and speed -> [0,1] then map to [-1,1]
nbor_rng = (jnp.clip(pos_norm / (sqrt3 + self.eps), 0.0, 1.0) * PRE(2.0) - PRE(1.0))
nbor_spd = (jnp.clip(vel_norm / (sqrt3 + self.eps), 0.0, 1.0) * PRE(2.0) - PRE(1.0))

# radial speed -> signed, scale to [-1,1]
r_hat = nbor_pos / (pos_norm[:, None] + self.eps)
radial = jnp.sum(r_hat * nbor_vel, axis=-1)  # (K,)
nbor_rdot = jnp.clip(radial / (sqrt3 + self.eps), -1.0, 1.0)

# tangential speed -> [0,1] then map to [-1,1]
u_perp = nbor_vel - radial[:, None] * r_hat
tang = jnp.linalg.norm(u_perp, axis=-1)
nbor_tan = (jnp.clip(tang / (sqrt3 + self.eps), 0.0, 1.0) * PRE(2.0) - PRE(1.0))

# Zero invariants for out-of-range neighbours (padding slots).
nbor_rng  = zero_1d(nbor_rng)
nbor_spd  = zero_1d(nbor_spd)
nbor_rdot = zero_1d(nbor_rdot)
nbor_tan  = zero_1d(nbor_tan)

# Stack invariants per neighbour: (K,4)
nbor_inv = jnp.stack([nbor_rng, nbor_spd, nbor_rdot, nbor_tan], axis=-1)

# ------------------------------------------------------------------
# Pairwise formation geometry: neighbour-neighbour distances (K choose 2 scalars).
# Compute on normalised positions, then scale by 2*sqrt(3) so max separation -> ~1.
present = ~oor_idxs
dmat = jnp.linalg.norm(nbor_pos[:, None, :] - nbor_pos[None, :, :], axis=-1)  # (K,K)
pair_d = dmat[self.pair_i, self.pair_j]  # (P,)
pair_m = present[self.pair_i] & present[self.pair_j]
pair_d = jnp.where(pair_m, pair_d, PRE(0.0))
pair_feat = (jnp.clip(pair_d / (PRE(2.0) * sqrt3 + self.eps), 0.0, 1.0) * PRE(2.0) - PRE(1.0))

nbor_tem = zero_1d(nbor_tem)

# Concatenate self attributes.
self_ = jnp.concatenate([self_pos, self_vel, self_att, self_fre, self_bat, self_act])

# Concatenate neighbour attributes and flatten.
nbor_ = jnp.concatenate([nbor_pos, nbor_vel, nbor_att,
                      nbor_fre[:, None], nbor_tem[:, None], nbor_inv], axis=-1).flatten()

# Append pairwise geometry scalars.
nbor_ = jnp.concatenate([nbor_, pair_feat], axis=-1)



class RayMix3(nn.Module):
    """Cheap 1D 'conv' over rays (axis=2) using depthwise kernel size 3.
    Input:  (..., R, S, C)
    Output: (..., R, S, C)
    """
    channels: int

    @nn.compact
    def __call__(self, x):
        # x: (..., R, S, C)
        C = x.shape[-1]
        assert C == self.channels, (C, self.channels)

        # Depthwise kernel weights: (3, C)
        k = self.param(
            "k",
            lambda key, shape: jnp.array(np.array([0.25, 0.5, 0.25], dtype=np.float32))[:, None]
                           * jnp.ones((1, shape[1]), dtype=jnp.float32),
            (3, C),
        )
        k0, k1, k2 = k[0], k[1], k[2]  # each (C,)
        up   = jnp.roll(x, shift=+1, axis=-3)
        down = jnp.roll(x, shift=-1, axis=-3)

        # broadcast (C,) across (..., R, S, C)
        out = (up   * k0) + (x * k1) + (down * k2)
        return out

# 1) cheap per-cell channel lift (this is GEMM-friendly)
ter = nn.Dense(16, use_bias=False, kernel_init=orthogonal(np.sqrt(2)))(ter)  # (B,N,view,view,16)

# 2) cheap "conv-like" spreading across rays only
ter = RayMix3(16, edge_mode="replicate")(ter)  # (B,N,view,view,16)

# 3) keep the rest identical
ter = self.act(ter).reshape(B, N, view, -1)
ter = nn.Dense(16, kernel_init=orthogonal(np.sqrt(2)))(ter)
ter = self.act(ter).reshape(B, N, -1)


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

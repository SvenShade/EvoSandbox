from typing import Callable, Sequence
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import lax


class FastConv9x9K3S1(nn.Module):
    """Specialized conv for inputs (..., 9, 9, C_in), kernel 3x3, stride 1, SAME.

    Returns (..., 9, 9, features). Parameters are learned like nn.Conv.

    This is typically faster than nn.Conv for very small spatial sizes because it:
      - extracts patches with lax.conv_general_dilated_patches
      - does one GEMM per call
    """

    features: int
    use_bias: bool = True
    dtype: jnp.dtype | None = jnp.bfloat16       # compute dtype (bf16 on GPU)
    param_dtype: jnp.dtype = jnp.float32         # store params fp32
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros
    precision: lax.Precision | None = None

    # Optional: pad the K dimension of GEMM to a multiple of 16 (sometimes helps bf16)
    pad_to: int | None = 16

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x)
        if x.shape[-3:-1] != (9, 9):
            raise ValueError(f"FastConv9x9K3S1 requires (..., 9, 9, C), got {x.shape}.")
        Cin = x.shape[-1]
        Cout = self.features

        # Flatten leading dims into one batch
        lead_shape = x.shape[:-3]
        if lead_shape:
            Bflat = int(np.prod(lead_shape))
            x2 = x.reshape(Bflat, 9, 9, Cin)
        else:
            x2 = x  # (B,9,9,Cin)

        compute_dtype = x2.dtype if self.dtype is None else self.dtype
        x2 = x2.astype(compute_dtype)

        # Extract patches: (Bflat, 9, 9, Cin*3*3) == (Bflat, 9, 9, 9*Cin)
        patches = lax.conv_general_dilated_patches(
            lhs=x2,
            filter_shape=(3, 3),
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            precision=self.precision,
        )
        K = 9 * Cin  # contracted dim

        # Weight matrix W: (K, Cout)
        W = self.param("kernel", self.kernel_init, (K, Cout)).astype(self.param_dtype)
        W = W.astype(compute_dtype)

        # Optional padding of K to help bf16 matmul kernels (try both in benchmarks)
        if self.pad_to is not None:
            Kpad = int(self.pad_to * ((K + self.pad_to - 1) // self.pad_to))
            if Kpad != K:
                patches = jnp.pad(patches, ((0, 0), (0, 0), (0, 0), (0, Kpad - K)))
                W = jnp.pad(W, ((0, Kpad - K), (0, 0)))
                K = Kpad

        # GEMM: reshape to 2D, multiply, reshape back
        # patches_2d: (Bflat*81, K), out_2d: (Bflat*81, Cout)
        patches_2d = patches.reshape(-1, K)
        out_2d = patches_2d @ W
        y2 = out_2d.reshape(x2.shape[0], 9, 9, Cout)

        if self.use_bias:
            b = self.param("bias", self.bias_init, (Cout,)).astype(self.param_dtype)
            y2 = y2 + b.astype(compute_dtype)

        # Restore leading dims
        if lead_shape:
            return y2.reshape(*lead_shape, 9, 9, Cout)
        return y2



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

"""
Set Transformer torso + rotation-equivariant continuous action head for SimpleSpread.

This file is intended to replace/augment your current `torsos.py` components when using:
  - fixed neighbour slots K=5 (padded with zeros)
  - invariant neighbour features (range, rel_speed, radial_speed, tangential_speed)
  - pairwise neighbour-neighbour distances (K*(K-1)//2 = 10) appended after neighbour slots
  - terrain patch (view x view), default view=9

High-level design (matches the plan you restated):
  1) Build *invariant scalar tokens* per neighbour slot (no raw xyz axes):
       inv(4) + fire(1) + team(1) + pairwise summaries(2)  -> token_in dim 8
     where pairwise summaries are per-neighbour min/mean distance to other neighbours
     computed from r (rotation-invariant, permutation-equivariant).

  2) Contextualize neighbour tokens with 1 masked Set Transformer block.

  3) Compute attention weights alpha_i from contextualized invariant tokens (+ ego context),
     with masked softmax. This makes alpha rotation-invariant and permutation-equivariant.

  4) Form equivariant vector summaries:
       v_r = Σ alpha_i r_i
       v_u = Σ alpha_i u_i
     which rotate correctly under global rotations.

  5) Pool scalar context c from the token set (PMA-style learned seed attention), embed
     terrain and the provided pairwise feature vector, and produce a scalar context
     embedding h.

  6) Output torso embedding as concat([h, v_r, v_u]) so the action head can produce
     a magnitude×direction mean with built-in rotation equivariance.

Modules provided:
  - SwarmSetEquivariantTorso: produces obs_embedding = [h, v_r, v_u] plus aux diagnostics
  - ContinuousActionHead: Mava-compatible action head (tanh-squashed Normal) with an
    equivariant mean computed from v_r/v_u and scalar gates from h.

Notes:
  - This torso does NOT include a value head; keep your existing critic head, or add one
    on top of `h` (recommended: value = Dense(1)(h)[...,0]).
  - Masking: padding slots are assumed to have r==0 (and other per-slot features == 0).
"""

from __future__ import annotations

from typing import Dict, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.initializers import orthogonal

# =========================
# Helper functions
# =========================

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import orthogonal


def masked_softmax(logits: jnp.ndarray, mask: jnp.ndarray, axis: int = -1, eps: float = 1e-9) -> jnp.ndarray:
    """Softmax over `axis` with boolean mask. Masked positions -> 0. Safe when all masked."""
    logits = jnp.asarray(logits)
    mask = mask.astype(bool)
    very_neg = jnp.array(-1e9, dtype=logits.dtype)
    masked_logits = jnp.where(mask, logits, very_neg)

    max_logits = jnp.max(masked_logits, axis=axis, keepdims=True)
    exp_logits = jnp.exp(masked_logits - max_logits) * mask.astype(logits.dtype)
    denom = jnp.sum(exp_logits, axis=axis, keepdims=True)
    return exp_logits / (denom + eps)


def safe_normalize(v: jnp.ndarray, axis: int = -1, eps: float = 1e-6) -> jnp.ndarray:
    n = jnp.linalg.norm(v, axis=axis, keepdims=True)
    return v / (n + eps)


def make_mask_safe(mask: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Ensure attention never sees an all-masked set.
    Returns: (mask_safe, mask_any) where
      mask_safe: same shape as mask, but if all False along K, slot 0 is forced True
      mask_any: (..., 1) boolean indicating whether any real elements exist
    """
    mask = mask.astype(bool)
    mask_any = jnp.any(mask, axis=-1, keepdims=True)  # (...,1)
    # Force slot 0 true for empty sets
    mask_safe = mask.at[..., 0].set(True)
    mask_safe = jnp.where(mask_any, mask, mask_safe)
    return mask_safe, mask_any


def zero_if_masked(x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Overwrite masked rows with 0 (safe even if x contains NaNs)."""
    return jnp.where(mask[..., None], x, jnp.array(0.0, dtype=x.dtype))


# =========================
# MLP
# =========================

class MLP(nn.Module):
    hidden: int
    out: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden, kernel_init=orthogonal(jnp.sqrt(2.0)), dtype=jnp.float32, param_dtype=jnp.float32)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden, kernel_init=orthogonal(jnp.sqrt(2.0)), dtype=jnp.float32, param_dtype=jnp.float32)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.out, kernel_init=orthogonal(jnp.sqrt(2.0)), dtype=jnp.float32, param_dtype=jnp.float32)(x)
        return x


# =========================
# Set Transformer Block (fixed masks + NaN-safe padding)
# =========================

class SetTransformerBlock(nn.Module):
    """Pre-norm transformer block over set tokens with padding mask.
    Supports x shaped (..., K, d_model) and mask shaped (..., K).
    """

    d_model: int
    num_heads: int
    mlp_hidden: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask_safe: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        """
        mask_safe: used ONLY for attention safety (never all-masked)
        mask:      semantic padding mask (true neighbors only) used to zero outputs
        """
        mask_safe = mask_safe.astype(bool)
        mask = mask.astype(bool)

        # Self-attn mask should be (..., 1, K, K) so it broadcasts to (..., H, K, K)
        qk = mask_safe[..., :, None] & mask_safe[..., None, :]        # (..., K, K)
        attn_mask = qk[..., None, :, :]                               # (..., 1, K, K)

        y = nn.LayerNorm(dtype=jnp.float32, param_dtype=jnp.float32)(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            dropout_rate=0.0,
            deterministic=True,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )(y, y, mask=attn_mask)

        # Critical: overwrite padded QUERY rows before residual add (avoid NaN * 0 leaks)
        y = zero_if_masked(y, mask)
        x = x + y

        z = nn.LayerNorm(dtype=jnp.float32, param_dtype=jnp.float32)(x)
        z = MLP(hidden=self.mlp_hidden, out=self.d_model)(z)
        z = zero_if_masked(z, mask)
        x = x + z

        # Final semantic cleanup
        x = zero_if_masked(x, mask)
        return x


# =========================
# AttentionPool (PMA-style; fixed masks + safe empty-set)
# =========================

class AttentionPool(nn.Module):
    """PMA-style pooling: learned seed query attends over set tokens.
    x: (..., K, d_model), mask_safe/mask: (..., K)
    returns: (..., d_model)
    """

    d_model: int
    num_heads: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask_safe: jnp.ndarray, mask_any: jnp.ndarray) -> jnp.ndarray:
        mask_safe = mask_safe.astype(bool)
        # mask_any: (...,1) indicates whether original set had any real members

        lead_shape = x.shape[:-2]  # (...)
        seed = self.param("seed", nn.initializers.normal(stddev=0.02), (1, 1, self.d_model))
        q = jnp.broadcast_to(seed, (*lead_shape, 1, self.d_model)).astype(jnp.float32)  # (...,1,d)

        # Cross-attn mask: (..., 1, 1, K) to broadcast to (..., H, 1, K)
        attn_mask = mask_safe[..., None, None, :]  # (...,1,1,K)

        pooled = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            dropout_rate=0.0,
            deterministic=True,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )(q, x, mask=attn_mask)  # (...,1,d)

        pooled = pooled[..., 0, :]  # (...,d)

        # If truly empty set, return zeros (avoid leaking arbitrary seed-attn output)
        pooled = jnp.where(mask_any, pooled, jnp.zeros_like(pooled))
        return pooled


# =========================
# SwarmSetEquivariantTorso (returns embedding; aux optionally)
# =========================

class SwarmSetEquivariantTorso(nn.Module):
    """
    Torso implementing:
      - invariant-token SetTx (formation reasoning)
      - alpha weights from invariant tokens + ego context
      - equivariant vector summaries v_r, v_u
      - scalar context h
    Returns obs_embedding = concat([h, v_r, v_u])
    """

    slots: int = 5
    per_slot_d: int = 15
    view: int = 9  # terrain is view*view
    d_model: int = 64
    num_heads: int = 4
    mlp_hidden: int = 128

    @nn.compact
    def __call__(self, obs: jnp.ndarray, return_aux: bool = False):
        obs = obs.astype(jnp.float32)
        B, N, D = obs.shape

        pair_d = self.slots * (self.slots - 1) // 2
        ter_d = self.view * self.view
        nbr_d = self.slots * self.per_slot_d + pair_d
        ego_d = D - nbr_d - ter_d
        if ego_d <= 0:
            raise ValueError(f"Inferred ego_d={ego_d} not positive. Check layout: D={D}, nbr_d={nbr_d}, ter_d={ter_d}")

        ego = obs[..., :ego_d]                               # (B,N,ego_d)
        nbr_all = obs[..., ego_d : ego_d + nbr_d]            # (B,N,nbr_d)
        ter = obs[..., -ter_d :].reshape(B, N, self.view, self.view, 1)  # (B,N,H,W,1) if you still use it elsewhere

        nbr_main = nbr_all[..., : self.slots * self.per_slot_d].reshape(B, N, self.slots, self.per_slot_d)
        pair = nbr_all[..., self.slots * self.per_slot_d :]  # (B,N,pair_d)

        # ---- split neighbor per-slot features
        r = nbr_main[..., 0:3]     # (B,N,K,3)
        u = nbr_main[..., 3:6]     # (B,N,K,3)
        att = nbr_main[..., 6:9]   # (B,N,K,3) <-- multi-objective attitude delta
        fre = nbr_main[..., 9:10]  # (B,N,K,1)
        tem = nbr_main[..., 10:11] # (B,N,K,1)
        inv = nbr_main[..., 11:15] # (B,N,K,4)

        # semantic padding mask: padding r==0 (your convention)
        mask = jnp.any(jnp.abs(r) > 1e-6, axis=-1)  # (B,N,K) bool

        # mask_safe prevents all-masked attention; mask_any used to zero pooled output
        mask_safe, mask_any = make_mask_safe(mask)

        # ---- pairwise summaries computed from r (robust; avoids pair ordering)
        r_clean = jnp.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        diff = r_clean[..., :, None, :] - r_clean[..., None, :, :]          # (B,N,K,K,3)
        dist = jnp.linalg.norm(diff, axis=-1)                               # (B,N,K,K)

        pair_mask = mask[..., :, None] & mask[..., None, :]                 # (B,N,K,K)
        K = self.slots
        eye = jnp.eye(K, dtype=bool)[None, None, :, :]                      # (1,1,K,K)
        valid = pair_mask & (~eye)                                          # exclude self

        big = jnp.array(1e9, dtype=dist.dtype)
        dist_no_self = jnp.where(eye, big, dist)

        d_min = jnp.min(jnp.where(valid, dist_no_self, big), axis=-1, keepdims=True)  # (B,N,K,1)
        d_sum = jnp.sum(jnp.where(valid, dist, 0.0), axis=-1, keepdims=True)          # (B,N,K,1)
        cnt = jnp.sum(valid, axis=-1, keepdims=True).astype(dist.dtype)               # (B,N,K,1)
        d_mean = jnp.where(cnt > 0, d_sum / (cnt + 1e-9), 0.0)                        # (B,N,K,1)

        # ---- invariant tokens (rotation-invariant geometry scalars + task scalars)
        # attitudes live in objective space (not geometric), safe to include
        att_norm = jnp.linalg.norm(att, axis=-1, keepdims=True)
        tok_in = jnp.concatenate([inv, fre, tem, att, att_norm, d_min, d_mean], axis=-1)  # (B,N,K, ?)

        # project tokens
        tok = nn.Dense(self.d_model, kernel_init=orthogonal(jnp.sqrt(2.0)), dtype=jnp.float32, param_dtype=jnp.float32)(tok_in)
        tok = nn.gelu(tok)
        tok = zero_if_masked(tok, mask)  # semantic cleanup

        # ---- SetTx formation cognition (attention uses mask_safe; output cleaned with mask)
        tok = SetTransformerBlock(d_model=self.d_model, num_heads=self.num_heads, mlp_hidden=self.mlp_hidden)(tok, mask_safe, mask)

        # ---- alpha logits from tok + ego context (broadcast ego_ctx to each slot)
        ego_ctx = MLP(hidden=self.mlp_hidden, out=self.d_model)(ego)                 # (B,N,d_model)
        ego_ctx_b = ego_ctx[..., None, :]                                            # (B,N,1,d_model) broadcasts in concat
        logits = MLP(hidden=self.mlp_hidden, out=1)(jnp.concatenate([tok, ego_ctx_b], axis=-1))[..., 0]  # (B,N,K)
        alpha = masked_softmax(logits, mask, axis=-1)                                # (B,N,K)

        # ---- equivariant vector summaries (only geometry vectors)
        alpha3 = alpha[..., None]
        v_r = jnp.sum(alpha3 * r, axis=-2)  # (B,N,3)
        v_u = jnp.sum(alpha3 * u, axis=-2)  # (B,N,3)

        # ---- pooled scalar context
        c = AttentionPool(d_model=self.d_model, num_heads=self.num_heads)(tok, mask_safe, mask_any)  # (B,N,d_model)

        # ---- include pairwise flat features too (since env provides them)
        pair_emb = nn.Dense(self.d_model, kernel_init=orthogonal(jnp.sqrt(2.0)), dtype=jnp.float32, param_dtype=jnp.float32)(pair)
        pair_emb = nn.gelu(pair_emb)

        # Scalar/context embedding h
        h_in = jnp.concatenate([ego, c, pair_emb], axis=-1)
        h = MLP(hidden=self.mlp_hidden, out=self.mlp_hidden)(h_in)  # (B,N,H)

        obs_embedding = jnp.concatenate([h, v_r, v_u], axis=-1)     # (B,N,H+6)

        if not return_aux:
            return obs_embedding

        aux = {
            "mask": mask,
            "mask_safe": mask_safe,
            "alpha": alpha,
            "logits": logits,
            "v_r": v_r,
            "v_u": v_u,
            "d_min": d_min[..., 0],
            "d_mean": d_mean[..., 0],
            "att_norm": att_norm[..., 0],
        }
        return obs_embedding, aux


# --------------------------
# Terrain encoder (same spirit as your existing torso)
# --------------------------

class FastConv3x3(nn.Module):
    """Fast 3x3 conv for inputs (..., H, W, 1) with circular padding on rows, zero on cols."""

    features: int
    use_bias: bool = True
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.float32
    kernel_init = nn.initializers.lecun_normal()
    bias_init = nn.initializers.zeros

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x)
        if x.shape[-1] != 1:
            raise ValueError(f"FastConv3x3 requires C_in=1, got {x.shape[-1]}.")
        H, W = x.shape[-3], x.shape[-2]
        lead = x.shape[:-3]

        if lead:
            Bflat = int(np.prod(lead))
            x2 = x.reshape(Bflat, H, W, 1)
        else:
            x2 = x
        x2 = x2.astype(self.dtype)

        # pad cols zero, rows circular
        xw = jnp.pad(x2, ((0, 0), (0, 0), (1, 1), (0, 0)))
        top = xw[:, -1:, :, :]
        bot = xw[:, :1, :, :]
        xp = jnp.concatenate([top, xw, bot], axis=1)

        base = xp[..., 0]
        shifts = [
            base[:, dy : dy + H, dx : dx + W]
            for dy in (0, 1, 2)
            for dx in (0, 1, 2)
        ]
        X9 = jnp.stack(shifts, axis=-1)  # (Bflat,H,W,9)

        Wmat = self.param("kernel", self.kernel_init, (9, self.features)).astype(self.param_dtype)
        Wmat = Wmat.astype(self.dtype)
        Y = (X9.reshape(-1, 9) @ Wmat).reshape(x2.shape[0], H, W, self.features)

        if self.use_bias:
            b = self.param("bias", self.bias_init, (self.features,)).astype(self.param_dtype)
            Y = Y + b.astype(self.dtype)

        if lead:
            return Y.reshape(*lead, H, W, self.features)
        return Y



class ContinuousActionHead(nn.Module):
    """Mava-compatible continuous action head with an equivariant mean.

    Expects obs_embedding = concat([h, v_r, v_u]) where:
      - h is scalar/context embedding (rotation-invariant content)
      - v_r, v_u are equivariant vector summaries in R^3 each

    Mean:
      loc = action_scale * sigmoid(mag(h)) * normalize( sigmoid(g_r,g_u)(h)[0]*v_r + ... )
    Std:
      - independent_std=True: single learned log_std per action dim (broadcast)
      - independent_std=False: log_std = Dense(h), depending only on h (keeps std rotation-stable)
    """

    action_dim: int
    min_scale: float = 1e-3
    independent_std: bool = True
    action_scale: float = 2.0  # scale before tanh

    def setup(self) -> None:
        if self.action_dim != 3:
            raise ValueError("This equivariant head is written for action_dim=3.")

        self.gates = nn.Dense(2, kernel_init=orthogonal(0.01))  # g_r, g_u in (0,1)
        self.mag = nn.Dense(1, kernel_init=orthogonal(0.01))    # magnitude in (0,1)

        if self.independent_std:
            self.log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        else:
            self.log_std = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))

    @nn.compact
    def __call__(self, obs_embedding: chex.Array, action_mask: chex.Array) -> tfd.Independent:
        del action_mask

        # Split embedding: last 6 dims are v_r (3) and v_u (3)
        h = obs_embedding[..., :-6]
        v_r = obs_embedding[..., -6:-3]
        v_u = obs_embedding[..., -3:]

        g = jax.nn.sigmoid(self.gates(h))
        g_r = g[..., 0:1]
        g_u = g[..., 1:2]

        v = g_r * v_r + g_u * v_u
        direction = safe_normalize(v)                 # (..,3)
        magnitude = jax.nn.sigmoid(self.mag(h))       # (..,1)

        loc = self.action_scale * magnitude * direction  # (..,3)

        # Std
        if self.independent_std:
            scale_raw = self.log_std * jnp.ones_like(loc)
        else:
            scale_raw = self.log_std(h)  # depend only on h (rotation-stable)
        scale = jax.nn.softplus(scale_raw) + self.min_scale

        distribution = tfd.Normal(loc=loc, scale=scale)
        return tfd.Independent(
            TanhTransformedDistribution(distribution),
            reinterpreted_batch_ndims=1,
        )

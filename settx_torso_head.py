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

# TFP-JAX for the transformed distribution in the action head
from tensorflow_probability.substrates import jax as tfp  # type: ignore
tfd = tfp.distributions
tfb = tfp.bijectors


# --------------------------
# Small utilities
# --------------------------

def masked_softmax(logits: jnp.ndarray, mask: jnp.ndarray, axis: int = -1, eps: float = 1e-9) -> jnp.ndarray:
    """Softmax over `axis` with boolean mask.

    logits: (..., K)
    mask:   (..., K) bool

    Returns probs where masked positions are 0. Safe when all masked (returns all zeros).
    """
    logits = jnp.asarray(logits)
    mask = mask.astype(bool)

    very_neg = jnp.array(-1e9, dtype=logits.dtype)
    masked_logits = jnp.where(mask, logits, very_neg)

    # stable exponentiation
    max_logits = jnp.max(masked_logits, axis=axis, keepdims=True)
    exp_logits = jnp.exp(masked_logits - max_logits) * mask.astype(logits.dtype)
    denom = jnp.sum(exp_logits, axis=axis, keepdims=True)
    return exp_logits / (denom + eps)


def safe_normalize(v: jnp.ndarray, axis: int = -1, eps: float = 1e-6) -> jnp.ndarray:
    n = jnp.linalg.norm(v, axis=axis, keepdims=True)
    return v / (n + eps)


class MLP(nn.Module):
    hidden: int
    out: int

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        x = nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)))(x)
        x = nn.gelu(x)
        x = nn.Dense(self.out, kernel_init=orthogonal(np.sqrt(2)))(x)
        return x


# --------------------------
# Set Transformer components
# --------------------------

class SetTransformerBlock(nn.Module):
    """Pre-norm transformer block over set tokens with correct padding mask.

    x:    (..., K, d_model)
    mask: (..., K) bool
    """

    d_model: int
    num_heads: int
    mlp_hidden: int

    @nn.compact
    def __call__(self, x: chex.Array, mask: chex.Array) -> chex.Array:
        mask = mask.astype(bool)

        # Correct self-attn mask: (..., K, K)
        attn_mask = mask[..., :, None] & mask[..., None, :]

        y = nn.LayerNorm()(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            dropout_rate=0.0,
            deterministic=True,
        )(y, y, mask=attn_mask)
        x = x + y

        z = nn.LayerNorm()(x)
        z = MLP(hidden=self.mlp_hidden, out=self.d_model)(z)
        x = x + z

        # Hard zero padded tokens so residuals can't leak
        x = x * mask.astype(x.dtype)[..., None]
        return x


class AttentionPool(nn.Module):
    """PMA-style pooling: a learned seed query attends over set tokens.

    x:    (..., K, d_model)
    mask: (..., K) bool
    returns: (..., d_model)
    """

    d_model: int
    num_heads: int

    @nn.compact
    def __call__(self, x: chex.Array, mask: chex.Array) -> chex.Array:
        mask = mask.astype(bool)

        lead_shape = x.shape[:-2]  # (...)
        seed = self.param("seed", nn.initializers.normal(stddev=0.02), (1, 1, self.d_model))
        q = jnp.broadcast_to(seed, (*lead_shape, 1, self.d_model))

        # Query length is 1: (..., 1, K)
        attn_mask = mask[..., None, :]

        pooled = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            dropout_rate=0.0,
            deterministic=True,
        )(q, x, mask=attn_mask)

        return pooled[..., 0, :]


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


# --------------------------
# Torso: produces embedding = [h, v_r, v_u]
# --------------------------

class SwarmSetEquivariantTorso(nn.Module):
    """Invariant-token SetTx torso + equivariant vector summaries.

    Input obs: (B,N,D) with trailing structure:
      [ ego (ego_d) , neighbour_main (K*per_slot_d) , pairwise (pair_d) , terrain (view*view) ]

    Returns:
      obs_embedding: (B,N, h_dim + 6) = concat([h, v_r, v_u])
      aux: diagnostics dict (alpha, v_r, v_u, mask, etc.)
    """

    # Observation constants
    slots: int = 5
    per_slot_d: int = 15
    view: int = 9

    # Model sizes
    d_model: int = 64
    num_heads: int = 4
    mlp_hidden: int = 128
    h_dim: int = 128  # scalar context embedding size (== mlp_hidden by default)

    @nn.compact
    def __call__(self, obs: chex.Array) -> Tuple[chex.Array, Dict[str, chex.Array]]:
        obs = obs.astype(jnp.float32)
        B, N, D = obs.shape

        pair_d = self.slots * (self.slots - 1) // 2
        ter_d = self.view * self.view
        nbr_d = self.slots * self.per_slot_d + pair_d
        ego_d = D - nbr_d - ter_d
        if ego_d <= 0:
            raise ValueError(
                f"Inferred ego_d={ego_d} is not positive. Check observation layout. "
                f"D={D}, nbr_d={nbr_d}, ter_d={ter_d}."
            )

        ego = obs[..., :ego_d]
        nbr_all = obs[..., ego_d : ego_d + nbr_d]
        ter = obs[..., -ter_d :].reshape(B, N, self.view, self.view, 1)

        nbr_main = nbr_all[..., : self.slots * self.per_slot_d].reshape(B, N, self.slots, self.per_slot_d)
        pair = nbr_all[..., self.slots * self.per_slot_d :]  # (B,N,pair_d)

        # ---- split neighbour per-slot features
        r = nbr_main[..., 0:3]    # (B,N,K,3)
        u = nbr_main[..., 3:6]    # (B,N,K,3)
        att = nbr_main[..., 6:9]  # (B,N,K,3)
        att_norm = jnp.linalg.norm(att, axis=-1, keepdims=True)  # (B,N,K,1)
        fre = nbr_main[..., 9:10]   # (B,N,K,1) in [-1,1]
        tem = nbr_main[..., 10:11]  # (B,N,K,1) (0 teammate, 1 opponent, 0 padding)
        inv = nbr_main[..., 11:15]  # (B,N,K,4) in [-1,1], padding=0

        # Padding mask (consistent with env: padded slots have r==0)
        mask = jnp.any(jnp.abs(r) > 1e-6, axis=-1)  # (B,N,K) bool

        # ---- per-neighbour pairwise summaries from r (rotation-invariant)
        # distance matrix: (B,N,K,K)
        diff = r[..., :, None, :] - r[..., None, :, :]
        d = jnp.linalg.norm(diff, axis=-1)

        # valid pairs for each i: neighbour i and j exist, and j != i
        eye = jnp.eye(self.slots, dtype=bool)
        eye = jnp.broadcast_to(eye, (B, N, self.slots, self.slots))
        valid = (mask[..., :, None] & mask[..., None, :]) & (~eye)

        # min distance to others (per i)
        huge = jnp.array(1e9, dtype=d.dtype)
        d_for_min = jnp.where(valid, d, huge)
        d_min = jnp.min(d_for_min, axis=-1)  # (B,N,K)
        # count of valid others (per i)
        cnt = jnp.sum(valid, axis=-1).astype(d.dtype)  # (B,N,K)
        d_min = jnp.where(cnt > 0, d_min, 0.0)

        # mean distance to others (per i)
        d_sum = jnp.sum(jnp.where(valid, d, 0.0), axis=-1)
        d_mean = jnp.where(cnt > 0, d_sum / (cnt + 1e-6), 0.0)

        # scale distances to [-1,1] to match env convention:
        # r in [-1,1]^3 => max pair distance is 2*sqrt(3)
        max_d = 2.0 * np.sqrt(3.0)
        d_min_s = jnp.clip(2.0 * (d_min / max_d) - 1.0, -1.0, 1.0)[..., None]   # (B,N,K,1)
        d_mean_s = jnp.clip(2.0 * (d_mean / max_d) - 1.0, -1.0, 1.0)[..., None] # (B,N,K,1)

        # ensure padded tokens have 0 summaries
        d_min_s = d_min_s * mask.astype(d_min_s.dtype)[..., None]
        d_mean_s = d_mean_s * mask.astype(d_mean_s.dtype)[..., None]

        # ---- invariant token set (rotation-invariant scalars + non-geom flags)
        tok_in = jnp.concatenate([inv, fre, tem, att, att_norm, d_min_s, d_mean_s], axis=-1)
        tok = nn.Dense(self.d_model, kernel_init=orthogonal(np.sqrt(2)))(tok_in)
        tok = nn.gelu(tok)
        tok = tok * mask.astype(tok.dtype)[..., None]

        # ---- Set Transformer block for formation cognition
        tok = SetTransformerBlock(d_model=self.d_model, num_heads=self.num_heads, mlp_hidden=self.mlp_hidden)(tok, mask)

        # ---- compute alpha logits from contextual tokens + ego context
        ego_ctx = MLP(hidden=self.mlp_hidden, out=self.d_model)(ego)  # (B,N,d_model)
        ego_ctx_b = jnp.broadcast_to(ego_ctx[..., None, :], (B, N, self.slots, self.d_model))

        logits = MLP(hidden=self.mlp_hidden, out=1)(jnp.concatenate([tok, ego_ctx_b], axis=-1))[..., 0]  # (B,N,K)
        alpha = masked_softmax(logits, mask, axis=-1)  # (B,N,K)

        # ---- equivariant vector summaries
        alpha3 = alpha[..., None]
        v_r = jnp.sum(alpha3 * r, axis=-2)  # (B,N,3)
        v_u = jnp.sum(alpha3 * u, axis=-2)  # (B,N,3)

        # ---- pooled scalar context from token set
        c = AttentionPool(d_model=self.d_model, num_heads=self.num_heads)(tok, mask)  # (B,N,d_model)

        # ---- terrain embedding
        t = FastConv3x3(16)(ter)
        t = nn.gelu(t).reshape(B, N, self.view, -1)
        t = nn.Dense(16, kernel_init=orthogonal(np.sqrt(2)))(t)
        t = nn.gelu(t).reshape(B, N, -1)  # (B,N,view*16)

        # ---- embed the provided pairwise formation scalars (10) as global context
        pair_emb = nn.Dense(self.d_model, kernel_init=orthogonal(np.sqrt(2)))(pair)
        pair_emb = nn.gelu(pair_emb)

        # ---- scalar context embedding h
        h_in = jnp.concatenate([ego, c, pair_emb, t], axis=-1)
        h = MLP(hidden=self.mlp_hidden, out=self.h_dim)(h_in)  # (B,N,h_dim)

        # ---- final embedding for the action head
        obs_embedding = jnp.concatenate([h, v_r, v_u], axis=-1)  # (B,N,h_dim+6)

        aux = {
            "mask": mask,
            "alpha": alpha,
            "logits": logits,
            "v_r": v_r,
            "v_u": v_u,
            "d_min": d_min_s[..., 0],
            "d_mean": d_mean_s[..., 0],
            "pair_emb": pair_emb,
        }
        return obs_embedding, aux


# --------------------------
# Action distribution: tanh-squashed Normal
# --------------------------

class TanhTransformedDistribution(tfd.TransformedDistribution):
    """Tanh-squashed distribution for actions in [-1,1]."""

    def __init__(self, distribution: tfd.Distribution):
        super().__init__(distribution=distribution, bijector=tfb.Tanh())


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

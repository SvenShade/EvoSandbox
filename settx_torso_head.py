"""Set-Transformer + equivariant vector-summary actor-critic for SimpleSpread.

This is a concrete MVP for your updated observation layout.

Assumed per-agent observation layout (concatenated):
  [ ego (ego_d) , neighbour_main (K*per_slot_d) , pairwise (pair_d) , terrain (view*view) ]

Where:
  K=5 neighbour slots (fixed, padded with zeros)
  per_slot_d=15 = [r(3), u(3), att(3), fire(1), team(1), inv(4)]
    - r: relative position / view_rad in [-1,1]
    - u: relative velocity / max_speed in ~[-1,1]
    - inv: [range, rel_speed, radial_speed, tangential_speed] in [-1,1] (padding=0)
  pair_d = K*(K-1)//2 = 10 neighbour-neighbour distances in [-1,1] (padding masked to 0)
  terrain view=9 -> 81 values, already in [-1,1]

Ego dimension ego_d is inferred as: D - (K*per_slot_d + pair_d) - terrain_d.

Outputs:
  mean: (B,N,3) in [-1,1]
  log_std: (B,N,3)
  value: (B,N)
  aux: dict of useful diagnostics (alpha, mag, dir, etc.)
"""

from __future__ import annotations

from typing import Dict, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.initializers import orthogonal


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


class SetTransformerBlock(nn.Module):
    """Pre-norm transformer block over set tokens with padding mask."""

    d_model: int
    num_heads: int
    mlp_hidden: int

    @nn.compact
    def __call__(self, x: chex.Array, mask: chex.Array) -> chex.Array:
        # x: (B,N,K,d_model) or (B,K,d_model)
        # mask: matching (B,N,K) or (B,K)
        attn_mask = mask[..., None, None, :]  # (...,1,1,K)

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

        # hard zero-out padded tokens so residuals can't leak
        x = x * mask.astype(x.dtype)[..., None]
        return x


class AttentionPool(nn.Module):
    """PMA-style pooling: a learned seed query attends over set tokens."""

    d_model: int
    num_heads: int

    @nn.compact
    def __call__(self, x: chex.Array, mask: chex.Array) -> chex.Array:
        # x: (B,N,K,d_model) or (B,K,d_model)
        # returns: (..., d_model)
        # create seed query with leading dims matching x excluding K
        lead_shape = x.shape[:-2]  # (...)
        K = x.shape[-2]

        seed = self.param("seed", nn.initializers.normal(stddev=0.02), (1, 1, self.d_model))

        # build q: (..., 1, d_model)
        q = jnp.broadcast_to(seed, (*lead_shape, 1, self.d_model))
        attn_mask = mask[..., None, None, :]  # (...,1,1,K)

        pooled = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            dropout_rate=0.0,
            deterministic=True,
        )(q, x, mask=attn_mask)

        return pooled[..., 0, :]  # (..., d_model)


# --------------------------
# Terrain encoder (kept close to your existing torso)
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
# Main module
# --------------------------

class SwarmSetEquivariantActorCritic(nn.Module):
    """Actor-critic implementing invariant-token SetTx + equivariant vector summaries.

    Outputs Gaussian parameters (mean/log_std) for a 3D continuous action.
    Mean is squashed to [-1,1] via tanh.
    """

    # Observation constants
    slots: int = 5
    per_slot_d: int = 15
    view: int = 9

    # Model sizes
    d_model: int = 64
    num_heads: int = 4
    mlp_hidden: int = 128

    # Action scaling before tanh
    action_scale: float = 2.0

    # Log-std clipping
    min_log_std: float = -5.0
    max_log_std: float = 2.0

    @nn.compact
    def __call__(self, obs: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array, Dict[str, chex.Array]]:
        """Forward.

        obs: (B,N,D)
        Returns:
          mean: (B,N,3) in [-1,1]
          log_std: (B,N,3)
          value: (B,N)
          aux: dict
        """
        obs = obs.astype(jnp.float32)
        B, N, D = obs.shape

        pair_d = self.slots * (self.slots - 1) // 2
        ter_d = self.view * self.view
        nbr_d = self.slots * self.per_slot_d + pair_d
        ego_d = D - nbr_d - ter_d
        if ego_d <= 0:
            raise ValueError(f"Inferred ego_d={ego_d} is not positive. Check observation layout. D={D}, nbr_d={nbr_d}, ter_d={ter_d}.")

        ego = obs[..., :ego_d]
        nbr_all = obs[..., ego_d : ego_d + nbr_d]
        ter = obs[..., -ter_d :].reshape(B, N, self.view, self.view, 1)

        nbr_main = nbr_all[..., : self.slots * self.per_slot_d].reshape(B, N, self.slots, self.per_slot_d)
        pair = nbr_all[..., self.slots * self.per_slot_d :]  # (B,N,pair_d)

        # ---- split neighbour per-slot features
        r = nbr_main[..., 0:3]   # (B,N,K,3)
        u = nbr_main[..., 3:6]   # (B,N,K,3)
        # att = nbr_main[..., 6:9]  # not used in invariant tokens in this MVP
        fre = nbr_main[..., 9:10]  # (B,N,K,1) already in [-1,1]
        tem = nbr_main[..., 10:11] # (B,N,K,1) team diff (0 teammate, 1 opponent, 0 padding)
        inv = nbr_main[..., 11:15] # (B,N,K,4) in [-1,1], padding=0

        # reconstruct padding mask from relative position (robust; padding r==0)
        mask = jnp.any(jnp.abs(r) > 1e-6, axis=-1)  # (B,N,K)

        # ---- invariant token set (rotation-invariant scalars + non-geom flags)
        tok_in = jnp.concatenate([inv, fre, tem], axis=-1)  # (B,N,K,6)
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

        # ---- terrain embedding (kept similar to your existing torso)
        t = FastConv3x3(16)(ter)
        t = nn.gelu(t).reshape(B, N, self.view, -1)
        t = nn.Dense(16, kernel_init=orthogonal(np.sqrt(2)))(t)
        t = nn.gelu(t).reshape(B, N, -1)  # (B,N,view*16)

        # ---- embed pairwise formation scalars
        pair_emb = nn.Dense(self.d_model, kernel_init=orthogonal(np.sqrt(2)))(pair)
        pair_emb = nn.gelu(pair_emb)

        # ---- scalar head state
        h_in = jnp.concatenate([ego, c, pair_emb, t], axis=-1)
        h = MLP(hidden=self.mlp_hidden, out=self.mlp_hidden)(h_in)  # (B,N,H)

        # gates for combining v_r and v_u (scalars in [0,1])
        gates = nn.Dense(2, kernel_init=orthogonal(np.sqrt(2)))(h)
        g = jax.nn.sigmoid(gates)
        g_r = g[..., 0:1]
        g_u = g[..., 1:2]

        v = g_r * v_r + g_u * v_u
        dir_vec = safe_normalize(v, axis=-1)

        mag = jax.nn.sigmoid(nn.Dense(1, kernel_init=orthogonal(np.sqrt(2)))(h))  # (B,N,1)

        pre_tanh = self.action_scale * mag * dir_vec
        mean = jnp.tanh(pre_tanh)  # (B,N,3)

        log_std = nn.Dense(3, kernel_init=orthogonal(0.01))(h)
        log_std = jnp.clip(log_std, self.min_log_std, self.max_log_std)

        value = nn.Dense(1, kernel_init=orthogonal(1.0))(h)[..., 0]  # (B,N)

        aux = {
            "alpha": alpha,
            "logits": logits,
            "v_r": v_r,
            "v_u": v_u,
            "g_r": g_r[..., 0],
            "g_u": g_u[..., 0],
            "mag": mag[..., 0],
            "dir": dir_vec,
            "pair_emb": pair_emb,
        }

        # NOTE: for equivariant head, return concat([h, v_r, v_u]),
        # where h is the scalar/context embedding, and v_r and v_u are equivariant vvector summaries
        return mean, log_std, value, aux



class ContinuousActionHead(nn.Module):
    """ContinuousActionHead using a transformed Normal distribution.

    NOTE: expects obs_embedding = concat([h, v_r, v_u]) where
      v_r, v_u are equivariant vector summaries in R^3 each.
    Actions lie in [-1, 1] via tanh transform.
    """

    action_dim: int
    min_scale: float = 1e-3
    independent_std: bool = True
    action_scale: float = 2.0  # scale before tanh

    def setup(self) -> None:
        # --- Structured mean head (equivariant) ---
        # These operate only on the scalar context h
        self.gates = nn.Dense(2, kernel_init=orthogonal(0.01))   # g_r, g_u in (0,1)
        self.mag = nn.Dense(1, kernel_init=orthogonal(0.01))     # magnitude in (0,1)

        # --- Std head stays the same as before ---
        if self.independent_std:
            self.log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        else:
            self.log_std = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))

    @nn.compact
    def __call__(self, obs_embedding: chex.Array, action_mask: chex.Array) -> tfd.Independent:
        del action_mask

        assert self.action_dim == 3, "This equivariant mean is written for 3D actions."

        # Split embedding: last 6 dims are v_r (3) and v_u (3)
        h = obs_embedding[..., :-6]
        v_r = obs_embedding[..., -6:-3]
        v_u = obs_embedding[..., -3:]

        g = jax.nn.sigmoid(self.gates(h))
        g_r = g[..., 0:1]
        g_u = g[..., 1:2]

        v = g_r * v_r + g_u * v_u
        direction = safe_normalize(v)                # (B,N,3), equivariant if v_r/v_u are
        magnitude = jax.nn.sigmoid(self.mag(h))      # (B,N,1), invariant scalar

        loc = self.action_scale * magnitude * direction  # (B,N,3)

        # Std (unchanged)
        if self.independent_std:
            scale_raw = self.log_std * jnp.ones_like(loc)
        else:
            scale_raw = self.log_std(h)  # important: depend only on h (scalars), not vectors
        scale = jax.nn.softplus(scale_raw) + self.min_scale

        distribution = tfd.Normal(loc=loc, scale=scale)
        return tfd.Independent(
            TanhTransformedDistribution(distribution),
            reinterpreted_batch_ndims=1,
        )


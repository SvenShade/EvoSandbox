import chex
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import orthogonal


def zero_masked(x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    while mask.ndim < x.ndim:
        mask = mask[..., None]
    return jnp.where(mask, x, 0.0)


def masked_softmax(logits: jnp.ndarray, mask: jnp.ndarray, axis: int) -> jnp.ndarray:
    """
    logits: (..., K) or (..., K, H)
    mask:   same leading dims as logits up to K, bool
    softmax only over valid entries; returns all zeros if all entries are masked.
    """
    mask = mask.astype(jnp.bool_)
    while mask.ndim < logits.ndim:
        mask = mask[..., None]

    neg_big = jnp.finfo(logits.dtype).min
    has_any = jnp.any(mask, axis=axis, keepdims=True)

    masked_logits = jnp.where(mask, logits, neg_big)
    masked_logits = jnp.where(has_any, masked_logits, 0.0)

    weights = nn.softmax(masked_logits, axis=axis)
    weights = jnp.where(mask, weights, 0.0)

    denom = jnp.sum(weights, axis=axis, keepdims=True)
    weights = jnp.where(denom > 0, weights / denom, 0.0)
    return weights


def safe_norm(x: jnp.ndarray, axis: int = -1, eps: float = 1e-6) -> jnp.ndarray:
    return jnp.sqrt(jnp.sum(jnp.square(x), axis=axis, keepdims=True) + eps)


class SetTxBlock(nn.Module):
    """One small masked self-attention block over invariant neighbor tokens."""
    d_model: int
    num_heads: int = 4
    mlp_mult: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        """
        x:    (B, N, K, D)
        mask: (B, N, K)
        """
        B, N, K, D = x.shape
        x = x.reshape(B * N, K, D)
        mask = mask.reshape(B * N, K)

        has_any = jnp.any(mask, axis=-1, keepdims=True)
        fallback = jnp.zeros_like(mask).at[:, 0].set(True)
        mask_safe = jnp.where(has_any, mask, fallback)

        attn_mask = nn.make_attention_mask(mask_safe, mask_safe, dtype=jnp.bool_)

        y = nn.LayerNorm()(x)
        y = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            kernel_init=orthogonal(jnp.sqrt(2.0)),
            out_kernel_init=orthogonal(0.01),
            deterministic=True,
        )(y, mask=attn_mask)
        y = zero_masked(y, mask)
        x = x + y

        y = nn.LayerNorm()(x)
        y = nn.Dense(self.mlp_mult * self.d_model, kernel_init=orthogonal(jnp.sqrt(2.0)))(y)
        y = nn.gelu(y)
        y = nn.Dense(self.d_model, kernel_init=orthogonal(0.01))(y)
        y = zero_masked(y, mask)
        x = x + y

        x = zero_masked(x, mask)
        return x.reshape(B, N, K, D)


class MaskedSoftmaxPool(nn.Module):
    """Learned masked softmax pooling over tokens."""
    d_model: int
    use_layernorm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        """
        x:    (B, N, K, D)
        mask: (B, N, K)
        returns: (B, N, D)
        """
        y = nn.LayerNorm()(x) if self.use_layernorm else x

        scores = nn.Dense(
            1,
            kernel_init=orthogonal(0.01),
            bias_init=nn.initializers.zeros,
        )(y)[..., 0]  # (B, N, K)

        log_temp = self.param("log_temp", nn.initializers.zeros, ())
        temp = jax.nn.softplus(log_temp) + 1e-3

        weights = masked_softmax(temp * scores, mask, axis=-1)  # (B, N, K)
        pooled = jnp.sum(weights[..., None] * x, axis=-2)
        return pooled


class EgoAdditivePool(nn.Module):
    """
    GATv2/Bahdanau-style additive ego-conditioned pool over tokens.
    This is the cleanest place to re-introduce ego-conditioned selection.
    """
    d_model: int
    hidden_dim: int | None = None

    @nn.compact
    def __call__(self, ego_q: jnp.ndarray, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        """
        ego_q: (B, N, Dq)
        x:     (B, N, K, D)
        mask:  (B, N, K)
        returns: (B, N, D)
        """
        hid = self.hidden_dim or self.d_model

        q = nn.Dense(hid, kernel_init=orthogonal(jnp.sqrt(2.0)))(ego_q)[:, :, None, :]  # (B,N,1,H)
        k = nn.Dense(hid, kernel_init=orthogonal(jnp.sqrt(2.0)))(x)                      # (B,N,K,H)

        e = nn.Dense(
            1,
            kernel_init=orthogonal(0.01),
            bias_init=nn.initializers.zeros,
        )(jnp.tanh(q + k))[..., 0]  # (B,N,K)

        alpha = masked_softmax(e, mask, axis=-1)
        pooled = jnp.sum(alpha[..., None] * x, axis=-2)
        return pooled


class InvariantVectorReadout(nn.Module):
    """
    Produce H invariant-derived attention heads alpha_i^(h),
    then form equivariant-lite vector summaries:
        v_r^(h) = sum_i alpha_i^(h) r_i
        v_u^(h) = sum_i alpha_i^(h) u_i
    """
    token_dim: int
    num_heads: int = 2
    use_ego_context: bool = True

    @nn.compact
    def __call__(
        self,
        inv_tok: jnp.ndarray,   # (B,N,K,D)
        ego_ctx: jnp.ndarray,   # (B,N,De)
        r: jnp.ndarray,         # (B,N,K,3)
        u: jnp.ndarray,         # (B,N,K,3)
        mask: jnp.ndarray,      # (B,N,K)
    ):
        B, N, K, D = inv_tok.shape

        if self.use_ego_context:
            ego_h = nn.Dense(self.token_dim, kernel_init=orthogonal(jnp.sqrt(2.0)))(ego_ctx)
            ego_h = ego_h[:, :, None, :].repeat(K, axis=2)
            h = jnp.concatenate([nn.LayerNorm()(inv_tok), ego_h], axis=-1)
        else:
            h = nn.LayerNorm()(inv_tok)

        h = nn.Dense(self.token_dim, kernel_init=orthogonal(jnp.sqrt(2.0)))(h)
        h = nn.gelu(h)

        logits = nn.Dense(
            self.num_heads,
            kernel_init=orthogonal(0.01),
            bias_init=nn.initializers.zeros,
        )(h)  # (B,N,K,H)

        alpha = masked_softmax(logits, mask, axis=-2)  # softmax over K

        v_r = jnp.einsum("bnkh,bnkd->bnhd", alpha, r)  # (B,N,H,3)
        v_u = jnp.einsum("bnkh,bnkd->bnhd", alpha, u)  # (B,N,H,3)

        v_r = v_r.reshape(B, N, 3 * self.num_heads)
        v_u = v_u.reshape(B, N, 3 * self.num_heads)
        return v_r, v_u, alpha


class MLPTorsoEquivariantLite(nn.Module):
    """
    Equivariant-lite torso:
      - ego/world features stay explicit
      - neighbor invariant scalars get SetTx + learned pooling
      - invariant-derived weights form vector summaries from raw r,u
      - old head interface stays unchanged

    Assumed obs layout:
        [ego | 5 * per_slot_d neighbor features | 10 pairwise features | terrain]

    Assumed per-slot layout:
        slot[..., 0:3]  = relative position r
        slot[..., 3:6]  = relative velocity u
        slot[..., 6:]   = non-geometric scalars / flags / type bits
    """

    slots: int = 5
    per_slot_d: int = 15
    view: int = 9

    inv_token_dim: int = 32
    hidden_dim: int = 128

    ego_hidden: int = 64
    pair_hidden: int = 32
    ter_hidden: int = 64

    settx_heads: int = 4
    vec_heads: int = 2

    use_ego_additive_pool: bool = True

    # Optional small slot-sensitive escape hatch over invariant tokens only.
    inv_flat_dim: int = 32
    inv_flat_scale: float = 0.25

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        x = x.astype(jnp.float32)
        B, N, D = x.shape

        pair_d = self.slots * (self.slots - 1) // 2
        ter_d = self.view * self.view
        nbr_d = self.slots * self.per_slot_d + pair_d
        ego_d = D - nbr_d - ter_d
        if ego_d <= 0:
            raise ValueError(
                f"Invalid inferred dims: ego_d={ego_d}, total D={D}, nbr_d={nbr_d}, ter_d={ter_d}"
            )

        ego = x[..., :ego_d]
        nbr_all = x[..., ego_d : ego_d + nbr_d]
        ter = x[..., -ter_d:]

        nbr = nbr_all[..., : self.slots * self.per_slot_d].reshape(B, N, self.slots, self.per_slot_d)
        pair = nbr_all[..., self.slots * self.per_slot_d :]

        # Padding mask: assumes padded slots are all-zero.
        mask = jnp.any(jnp.abs(nbr) > 1e-6, axis=-1)  # (B,N,K)

        # -----------------------------
        # Ego / world stream (unchanged)
        # -----------------------------
        ego_e = nn.gelu(nn.Dense(self.ego_hidden, kernel_init=orthogonal(jnp.sqrt(2.0)))(ego))
        pair_e = nn.gelu(nn.Dense(self.pair_hidden, kernel_init=orthogonal(jnp.sqrt(2.0)))(pair))
        ter_e = nn.gelu(nn.Dense(self.ter_hidden, kernel_init=orthogonal(jnp.sqrt(2.0)))(ter))

        # ----------------------------------------
        # Split neighbor slot into vector + scalar
        # ----------------------------------------
        r = nbr[..., 0:3]
        u = nbr[..., 3:6] if self.per_slot_d >= 6 else jnp.zeros_like(r)
        extra = nbr[..., 6:]  # non-geometric scalars/flags/type bits

        rho = safe_norm(r)                        # |r|
        sigma = safe_norm(u)                      # |u|
        r_hat = r / rho
        radial = jnp.sum(r_hat * u, axis=-1, keepdims=True)
        tangential = safe_norm(u - radial * r_hat)

        # Invariant scalar token per neighbor.
        inv = jnp.concatenate([rho, sigma, radial, tangential, extra], axis=-1)
        inv = zero_masked(inv, mask)

        # ---------------------------------------------------
        # Invariant token encoder (this is the SetTx "brains")
        # ---------------------------------------------------
        inv_tok = nn.Dense(
            self.inv_token_dim,
            kernel_init=orthogonal(jnp.sqrt(2.0)),
        )(inv)
        inv_tok = nn.gelu(inv_tok)
        inv_tok = zero_masked(inv_tok, mask)

        inv_tok = SetTxBlock(
            d_model=self.inv_token_dim,
            num_heads=self.settx_heads,
        )(inv_tok, mask)

        # ---------------------------------
        # Invariant summaries / readout
        # ---------------------------------
        inv_sum = jnp.sum(zero_masked(inv_tok, mask), axis=-2)
        inv_cnt = jnp.maximum(jnp.sum(mask, axis=-1, keepdims=True), 1)
        inv_mean = inv_sum / inv_cnt

        neg_big = jnp.full_like(inv_tok, -1e9)
        inv_max = jnp.max(jnp.where(mask[..., None], inv_tok, neg_big), axis=-2)
        inv_max = jnp.where(jnp.any(mask, axis=-1, keepdims=True), inv_max, 0.0)

        inv_soft = MaskedSoftmaxPool(d_model=self.inv_token_dim)(inv_tok, mask)

        if self.use_ego_additive_pool:
            inv_add = EgoAdditivePool(d_model=self.inv_token_dim)(ego_e, inv_tok, mask)
        else:
            inv_add = jnp.zeros_like(inv_soft)

        # Optional tiny slot-sensitive path over invariant tokens only.
        z_parts = [ego_e, pair_e, ter_e, inv_mean, inv_max, inv_soft, inv_add]

        if self.inv_flat_dim > 0:
            inv_flat = inv_tok.reshape(B, N, self.slots * self.inv_token_dim)
            inv_flat = nn.Dense(self.inv_flat_dim, kernel_init=orthogonal(jnp.sqrt(2.0)))(inv_flat)
            inv_flat = nn.gelu(inv_flat)
            inv_flat = self.inv_flat_scale * inv_flat
            z_parts.append(inv_flat)

        # ----------------------------------------------------
        # Equivariant-lite vector summaries from invariant alphas
        # ----------------------------------------------------
        v_r, v_u, _alpha = InvariantVectorReadout(
            token_dim=self.inv_token_dim,
            num_heads=self.vec_heads,
            use_ego_context=True,
        )(inv_tok, ego_e, r, u, mask)

        z_parts.extend([v_r, v_u])

        # Final scalar fusion MLP; old head can consume this unchanged.
        z = jnp.concatenate(z_parts, axis=-1)

        z = nn.Dense(self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2.0)))(z)
        z = nn.gelu(z)
        z = nn.LayerNorm()(z)

        z = nn.Dense(self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2.0)))(z)
        z = nn.gelu(z)
        z = nn.LayerNorm()(z)

        return z

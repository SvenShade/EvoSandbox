import chex
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import orthogonal


def zero_masked(x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(mask[..., None], x, 0.0)


def masked_softmax(logits: jnp.ndarray, mask: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """
    logits: (..., K)
    mask:   (..., K) bool
    returns weights summing to 1 over valid entries, or all zeros if no valid entries
    """
    neg_big = jnp.finfo(logits.dtype).min
    has_any = jnp.any(mask, axis=axis, keepdims=True)

    masked_logits = jnp.where(mask, logits, neg_big)
    masked_logits = jnp.where(has_any, masked_logits, 0.0)

    weights = nn.softmax(masked_logits, axis=axis)
    weights = jnp.where(mask, weights, 0.0)

    denom = jnp.sum(weights, axis=axis, keepdims=True)
    weights = jnp.where(denom > 0, weights / denom, 0.0)
    return weights


class SetTxBlock(nn.Module):
    """One tiny padding-safe self-attention block over K neighbor slots."""
    d_model: int
    num_heads: int = 4
    mlp_mult: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        """
        x:    (B, N, K, d_model)
        mask: (B, N, K) bool, True for real neighbor
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
    """
    Learned masked softmax pooling over neighbor tokens.

    This is the main new patch:
      scores_k = f(token_k)
      alpha = softmax(temp * scores_k) over valid slots only
      pooled = sum_k alpha_k * token_k
    """
    d_model: int
    use_layernorm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        """
        x:    (B, N, K, d_model)
        mask: (B, N, K) bool
        returns: (B, N, d_model)
        """
        y = nn.LayerNorm()(x) if self.use_layernorm else x

        scores = nn.Dense(
            1,
            kernel_init=orthogonal(0.01),
            bias_init=nn.initializers.zeros,
        )(y)[..., 0]  # (B, N, K)

        # Learnable inverse temperature, initialized near 1.0
        log_temp = self.param("log_temp", nn.initializers.zeros, ())
        temp = nn.softplus(log_temp) + 1e-3

        weights = masked_softmax(temp * scores, mask, axis=-1)  # (B, N, K)
        pooled = jnp.sum(weights[..., None] * x, axis=-2)       # (B, N, d_model)
        return pooled


class MLPTorsoSetTxSoftPool(nn.Module):
    """
    Old-style MLP torso with one minimal SetTx insertion + learned softmax pooling.

    Assumed obs layout:
        [ego | 5 * per_slot_d neighbor features | 10 pairwise features | terrain]

    where:
        per_slot_d = 15
        pairwise features = C(5, 2) = 10
        terrain = view * view
    """

    slots: int = 5
    per_slot_d: int = 15
    view: int = 9

    nbr_token_dim: int = 24
    hidden_dim: int = 128
    ego_hidden: int = 64
    pair_hidden: int = 32
    ter_hidden: int = 64

    settx_heads: int = 4

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

        nbr = nbr_all[..., : self.slots * self.per_slot_d].reshape(
            B, N, self.slots, self.per_slot_d
        )
        pair = nbr_all[..., self.slots * self.per_slot_d :]

        # Mask padded neighbor slots. This assumes padding slots are all-zero.
        mask = jnp.any(jnp.abs(nbr) > 1e-6, axis=-1)  # (B, N, K)

        # Neighbor tokens -> contextualize with one SetTx block.
        nbr_tok = nn.Dense(
            self.nbr_token_dim,
            kernel_init=orthogonal(jnp.sqrt(2.0)),
        )(nbr)
        nbr_tok = nn.gelu(nbr_tok)
        nbr_tok = zero_masked(nbr_tok, mask)

        nbr_tok = SetTxBlock(
            d_model=self.nbr_token_dim,
            num_heads=self.settx_heads,
        )(nbr_tok, mask)

        # Existing summaries retained.
        nbr_flat = nbr_tok.reshape(B, N, self.slots * self.nbr_token_dim)

        nbr_sum = jnp.sum(zero_masked(nbr_tok, mask), axis=-2)
        nbr_cnt = jnp.maximum(jnp.sum(mask, axis=-1, keepdims=True), 1)
        nbr_mean = nbr_sum / nbr_cnt

        neg_big = jnp.full_like(nbr_tok, -1e9)
        nbr_max = jnp.max(jnp.where(mask[..., None], nbr_tok, neg_big), axis=-2)
        nbr_max = jnp.where(jnp.any(mask, axis=-1, keepdims=True), nbr_max, 0.0)

        # New learned readout.
        nbr_soft = MaskedSoftmaxPool(d_model=self.nbr_token_dim)(nbr_tok, mask)

        # Keep the rest of the torso old/simple.
        ego_e = nn.gelu(nn.Dense(self.ego_hidden, kernel_init=orthogonal(jnp.sqrt(2.0)))(ego))
        pair_e = nn.gelu(nn.Dense(self.pair_hidden, kernel_init=orthogonal(jnp.sqrt(2.0)))(pair))
        ter_e = nn.gelu(nn.Dense(self.ter_hidden, kernel_init=orthogonal(jnp.sqrt(2.0)))(ter))

        z = jnp.concatenate(
            [ego_e, pair_e, ter_e, nbr_flat, nbr_mean, nbr_max, nbr_soft],
            axis=-1,
        )

        z = nn.Dense(self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2.0)))(z)
        z = nn.gelu(z)
        z = nn.LayerNorm()(z)

        z = nn.Dense(self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2.0)))(z)
        z = nn.gelu(z)
        z = nn.LayerNorm()(z)

        return z

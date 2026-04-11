# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Dict, Sequence

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.initializers import orthogonal


def zero_masked(x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(mask[..., None], x, 0.0)


class SetTxBlock(nn.Module):
    """Padding-safe self-attention block over K neighbor slots."""
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

        # Avoid all-masked rows.
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


class EgoCrossAttnPool(nn.Module):
    """Ego-query attention pooling over contextualized neighbor tokens."""
    d_model: int
    num_heads: int = 4

    @nn.compact
    def __call__(self, ego_q: jnp.ndarray, nbr_tok: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        """
        ego_q:   (B, N, d_model)
        nbr_tok: (B, N, K, d_model)
        mask:    (B, N, K) bool, True for real neighbor
        """
        B, N, K, D = nbr_tok.shape

        q = ego_q.reshape(B * N, 1, D)
        kv = nbr_tok.reshape(B * N, K, D)
        mask = mask.reshape(B * N, K)

        has_any = jnp.any(mask, axis=-1, keepdims=True)               # (BN, 1)
        fallback = jnp.zeros_like(mask).at[:, 0].set(True)            # (BN, K)
        mask_safe = jnp.where(has_any, mask, fallback)

        q_valid = jnp.ones((B * N, 1), dtype=jnp.bool_)
        attn_mask = nn.make_attention_mask(q_valid, mask_safe, dtype=jnp.bool_)

        y = nn.LayerNorm()(q)
        kv = nn.LayerNorm()(kv)

        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            kernel_init=orthogonal(jnp.sqrt(2.0)),
            out_kernel_init=orthogonal(0.01),
            deterministic=True,
        )(y, kv, mask=attn_mask)

        y = y[:, 0, :]                                # (BN, D)
        y = jnp.where(has_any, y, 0.0)               # zero if no real neighbors
        return y.reshape(B, N, D)


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


class EgoAdditivePool(nn.Module):
    """
    Bahdanau-style additive ego-conditioned pool over tokens.
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

        q = nn.Dense(hid, kernel_init=orthogonal(jnp.sqrt(2.0)))(ego_q)[:, :, None, :]
        k = nn.Dense(hid, kernel_init=orthogonal(jnp.sqrt(2.0)))(x)

        e = nn.Dense(
            1,
            kernel_init=orthogonal(0.01),
            bias_init=nn.initializers.zeros,
        )(jnp.tanh(q + k))[..., 0]  # (B, N, K)

        alpha = masked_softmax(e, mask, axis=-1)
        pooled = jnp.sum(alpha[..., None] * x, axis=-2)
        return pooled


class InvariantVectorReadout(nn.Module):
    """
    Build attention heads from invariant tokens, then form weighted sums of raw vectors.
    """
    token_dim: int
    num_heads: int = 2
    use_ego_context: bool = True

    @nn.compact
    def __call__(
        self,
        inv_tok: jnp.ndarray,   # (B, N, K, D)
        ego_ctx: jnp.ndarray,   # (B, N, De)
        r: jnp.ndarray,         # (B, N, K, 3)
        u: jnp.ndarray,         # (B, N, K, 3)
        mask: jnp.ndarray,      # (B, N, K)
    ):
        B, N, K, _ = inv_tok.shape

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
        )(h)  # (B, N, K, H)

        alpha = masked_softmax(logits, mask[..., None], axis=-2)  # softmax over K

        v_r = jnp.einsum("bnkh,bnkd->bnhd", alpha, r)  # (B, N, H, 3)
        v_u = jnp.einsum("bnkh,bnkd->bnhd", alpha, u)  # (B, N, H, 3)

        v_r = v_r.reshape(B, N, 3 * self.num_heads)
        v_u = v_u.reshape(B, N, 3 * self.num_heads)
        return v_r, v_u, alpha


class DenseSeq(nn.Module):
    hidden: int = 64
    act_str: str = "gelu"
    
    def setup(self) -> None:
        self.act = _parse_activation_fn(self.act_str)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden, kernel_init=orthogonal(jnp.sqrt(2.0)))(x)
        x = nn.LayerNorm()(x)
        x = self.act(x)
        return x


class Bilinear(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x1, x2):
        # x1: [batch, n], x2: [batch, m], weights: [n, m, features]
        kernel = self.param('kernel',
                            nn.initializers.glorot_uniform(),
                            (x1.shape[-1], x2.shape[-1], self.features))
        
        # Return bilinear product.
        return jnp.einsum('...i,...j,ijk->...k', x1, x2, kernel)


class NbrEmbedFull(nn.Module):
    tok_emb: int = 24
    out_dim: int = 48
    num_heads: int = 3

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray, ego: jnp.ndarray) -> jnp.ndarray:
        # Set Transformer block.
        x = SetTxBlock(d_model=self.tok_emb, num_heads=self.num_heads)(x, mask)

        # Learned pooling stages.
        nbr_soft = MaskedSoftmaxPool(d_model=self.tok_emb)(x, mask)
        nbr_add  = EgoAdditivePool(d_model=self.tok_emb)(ego, x, mask)
        ego_q    = nn.Dense(self.tok_emb, kernel_init=orthogonal(jnp.sqrt(2.0)))(ego)
        nbr_attn = EgoCrossAttnPool(
            d_model=self.tok_emb,
            num_heads=self.num_heads,
        )(ego_q, x, mask)

        # Combine pooling and return.
        x = DenseSeq(self.out_dim)(jnp.concatenate([nbr_soft, nbr_add, nbr_attn], axis=-1))        
        return x


class NbrEmbedBase(nn.Module):
    tok_emb: int = 24
    out_dim: int = 48

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        # Basic neighbour pooling stages.
        B,N,K,D  = x.shape
        nbr_flat = x.reshape(B, N, K * D)
        nbr_sum  = jnp.sum(zero_masked(x, mask), axis=-2)
        nbr_cnt  = jnp.maximum(jnp.sum(mask, axis=-1, keepdims=True), 1)
        nbr_mean = nbr_sum / nbr_cnt
        nbr_max  = jnp.where(
            jnp.any(mask, axis=-1, keepdims=True),
            jnp.max(jnp.where(mask[..., None], x, jnp.full_like(x, -1e9)), axis=-2),
            0.0,
        )

        # Combine pooling and return.
        x = DenseSeq(self.out_dim)(jnp.concatenate([nbr_flat, nbr_mean, nbr_max], axis=-1))        
        return x


class MLPTorso(nn.Module):
    """
    Assumed obs layout for the patched simple_spread environment:
        [ego | k * nbr_slot_d | k * pgon_slot_d | C(k, 2) pairwise | terrain]

    where:
        ego is inferred
        k (reserved slots) = 5
        nbr_slot_d (agent-neighbour features) = 15
        pgon_slot_d (clay-pigeon features) = 11
        pairwise features = C(k, 2) = 10
        terrain = view (9 pixels) ** 2

    For backwards compatibility, this torso also accepts the older layout
    without the clay-pigeon block:
        [ego | k * nbr_slot_d | C(k, 2) pairwise | terrain]
    """
    # NOTE: THE FOLLOWING VARIABLES ARE NOMINAL/INACTIVE, HERE TO SATISFY mlp.yaml
    layer_sizes: Sequence[int]
    activation: str = "relu"
    use_layer_norm: bool = False
    activate_final: bool = True

    k: int = 5
    nbr_slot_d: int = 15
    pgon_slot_d: int = 11
    view: int = 9
    out_dim: int = 128
    ego_emb: int = 48
    nbr_emb: int = 48
    pgon_emb: int = 32
    ter_emb: int = 24
    nbr_tok_emb: int = 24
    pgon_tok_emb: int = 16
    pair_emb: int = 24
    ray_emb: int = 16
    settx_heads: int = 4

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        x = x.astype(jnp.float32)
        B, N, D = x.shape

        pair_d = self.k * (self.k - 1) // 2
        ter_d = self.view * self.view
        nbr_block_d = self.k * self.nbr_slot_d
        pgon_block_d = self.k * self.pgon_slot_d

        old_obs_d = 19 + nbr_block_d + pair_d + ter_d
        new_obs_d = 19 + nbr_block_d + pgon_block_d + pair_d + ter_d

        if D == new_obs_d:
            ego_d = D - nbr_block_d - pgon_block_d - pair_d - ter_d
            ego = x[..., :ego_d]
            off = ego_d
            nbr = x[..., off : off + nbr_block_d].reshape(B, N, self.k, self.nbr_slot_d)
            off += nbr_block_d
            pgon = x[..., off : off + pgon_block_d].reshape(B, N, self.k, self.pgon_slot_d)
            off += pgon_block_d
            pair = x[..., off : off + pair_d]
            ter = x[..., off + pair_d :].reshape(B, N, self.view, self.view)
        elif D == old_obs_d:
            ego_d = D - nbr_block_d - pair_d - ter_d
            ego = x[..., :ego_d]
            off = ego_d
            nbr = x[..., off : off + nbr_block_d].reshape(B, N, self.k, self.nbr_slot_d)
            off += nbr_block_d
            pair = x[..., off : off + pair_d]
            ter = x[..., off + pair_d :].reshape(B, N, self.view, self.view)
            pgon = jnp.zeros((B, N, self.k, self.pgon_slot_d), dtype=x.dtype)
        else:
            raise ValueError(
                f"MLPTorso expected obs dim {old_obs_d} (old) or {new_obs_d} (with pigeons), got {D}."
            )

        nbr_r = nbr[..., 0:3]
        pgon_r = pgon[..., 0:3]

        # Embed ego, pairs, and terrain.
        ego_e = DenseSeq(self.ego_emb)(ego)
        pair_e = DenseSeq(self.pair_emb)(pair)
        ter_e = DenseSeq(self.ter_emb)(DenseSeq(self.ray_emb)(ter).reshape(B, N, -1))

        # Padding masks from relative position channels.
        nbr_mask = jnp.any(jnp.abs(nbr_r) > 1e-6, axis=-1)
        pgon_mask = jnp.any(jnp.abs(pgon_r) > 1e-6, axis=-1)

        # Neighbour tokens.
        nbr_tok = DenseSeq(self.nbr_tok_emb)(nbr)
        nbr_tok = zero_masked(nbr_tok, nbr_mask)

        # Clay-pigeon tokens.
        pgon_tok = DenseSeq(self.pgon_tok_emb)(pgon)
        pgon_tok = zero_masked(pgon_tok, pgon_mask)

        # Split neighbour torso. Process teams differently.
        nbr_tok_0 = NbrEmbedBase(
            tok_emb=self.nbr_tok_emb,
            out_dim=self.nbr_emb,
        )(nbr_tok[:, : N // 2], nbr_mask[:, : N // 2])
        nbr_tok_1 = NbrEmbedFull(
            tok_emb=self.nbr_tok_emb,
            out_dim=self.nbr_emb,
            num_heads=self.settx_heads,
        )(nbr_tok[:, N // 2 :], nbr_mask[:, N // 2 :], ego_e[:, N // 2 :])
        nbr_e = jnp.concatenate([nbr_tok_0, nbr_tok_1], axis=1)

        # Use the same masked token machinery for clay pigeons, but symmetrically:
        # they are targets rather than teammates, so no team-split processing.
        pgon_e = NbrEmbedFull(
            tok_emb=self.pgon_tok_emb,
            out_dim=self.pgon_emb,
            num_heads=max(1, self.settx_heads // 2),
        )(pgon_tok, pgon_mask, ego_e)

        # Concatenate and pass through output layer.
        x = jnp.concatenate([ego_e, pair_e, nbr_e, pgon_e, ter_e], axis=-1)
        x = DenseSeq(self.out_dim)(x)
        return x


class MLPTorsoOG(nn.Module):
    """MLP torso."""

    layer_sizes: Sequence[int]
    activation: str = "relu"
    use_layer_norm: bool = False
    activate_final: bool = True

    def setup(self) -> None:
        self.activation_fn = _parse_activation_fn(self.activation)

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        """Forward pass."""
        x = observation.astype(jnp.bfloat16)
        for i, layer_size in enumerate(self.layer_sizes):
            x = nn.Dense(layer_size, kernel_init=orthogonal(np.sqrt(2)))(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(use_scale=False)(x)

            should_activate = (i < len(self.layer_sizes) - 1) or self.activate_final
            x = self.activation_fn(x) if should_activate else x

        return x


class CNNTorso(nn.Module):
    """CNN torso."""

    channel_sizes: Sequence[int]
    kernel_sizes: Sequence[int]
    strides: Sequence[int]
    activation: str = "relu"
    use_layer_norm: bool = False

    def setup(self) -> None:
        self.activation_fn = _parse_activation_fn(self.activation)

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        """Forward pass."""
        x = observation
        for channel, kernel, stride in zip(
            self.channel_sizes, self.kernel_sizes, self.strides, strict=True
        ):
            x = nn.Conv(channel, (kernel, kernel), (stride, stride))(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(use_scale=False)(x)
            x = self.activation_fn(x)

        # Collapse (merge) the last three dimensions (width, height, channels)
        # Leave the batch, agent and time (if recurrent) dims unchanged.
        return jax.lax.collapse(x, -3)


class SwiGLU(nn.Module):
    """SwiGLU module.
    A gated variation of a standard feedforward layer using a Swish activation function.
    For more details see: https://arxiv.org/abs/2002.05202
    """

    hidden_dim: int
    embed_dim: int

    def setup(self) -> None:
        self.W_linear = self.param(
            "W_linear", nn.initializers.zeros, (self.embed_dim, self.hidden_dim)
        )
        self.W_gate = self.param("W_gate", nn.initializers.zeros, (self.embed_dim, self.hidden_dim))
        self.W_output = self.param(
            "W_output", nn.initializers.zeros, (self.hidden_dim, self.embed_dim)
        )

    def __call__(self, x: chex.Array) -> chex.Array:
        gated_output = jax.nn.swish(x @ self.W_gate) * (x @ self.W_linear)
        return gated_output @ self.W_output


def _parse_activation_fn(activation_fn_name: str) -> Callable[[chex.Array], chex.Array]:
    """Get the activation function."""
    activation_fns: Dict[str, Callable[[chex.Array], chex.Array]] = {
        "relu": nn.relu,
        "gelu": nn.gelu,
        "tanh": nn.tanh,
    }
    return activation_fns[activation_fn_name]

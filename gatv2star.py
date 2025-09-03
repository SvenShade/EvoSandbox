"""
JAX/Flax implementation of a fast Star-Graph GATv2 encoder + policy head,
porting the structure of the provided PyTorch/torch_geometric module.

Key design choices:
- We exploit the specific graph structure used in the PyTorch code: a STAR graph
  per batch item (all nodes -> central node 0) with self-loops. This lets us
  avoid general sparse ops and implement a dense, vectorised attention pass.
- Two GATv2 layers with multi-head attention (concat over heads), GELU+LayerNorm
  after each conv (matching fc_node_norm / conv*_norm), then a learnable
  SoftmaxAggregation over nodes, followed by a linear + GELU + LayerNorm head,
  and finally a SlimFC (normc init + tanh) policy branch with optional
  "simple_norm".

Inputs:
- `inputs`: [B, N * F] or [B, N, F]. If `local=True`, the last feature is assumed
  to be a visibility flag; we zero out unseen nodes by multiplying by it.

Outputs:
- Encoder returns [B, output_dim]. If you instantiate GraphPolicy, you also get
  action logits via a SlimFC with tanh (mirroring the original SlimFC design).

Dependencies:
- jax, flax.linen (no jraph needed)

Performance:
- Fully batched and JIT-friendly; no Python loops over nodes or heads.
"""
from __future__ import annotations
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn


# ------------------------------- Utilities ---------------------------------

def gelu(x: jnp.ndarray) -> jnp.ndarray:
    return nn.gelu(x, approximate=False)


def normc_initializer(std: float = 1.0):
    """Column-normalized initialization as in RL literature (normc).

    For a weight matrix W of shape (in_dim, out_dim), we draw N(0,1) and then
    scale columns to have L2 norm = `std`.
    """
    def init(key: jax.Array, shape: Tuple[int, ...], dtype=jnp.float32):
        W = jax.random.normal(key, shape, dtype)
        if len(shape) == 2:
            # Normalize columns
            col_norm = jnp.linalg.norm(W, axis=0, keepdims=True) + 1e-8
            W = W / col_norm * std
            return W
        # For non-2D shapes (e.g., biases), fall back to Xavier uniform.
        fan_in = shape[0] if shape else 1
        lim = jnp.sqrt(6.0 / fan_in)
        return jax.random.uniform(key, shape, dtype, minval=-lim, maxval=lim)
    return init


class LayerNorm1d(nn.Module):
    """LayerNorm over the last dimension (channels)."""
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.LayerNorm(epsilon=self.epsilon, use_bias=True, use_scale=True)(x)


# ------------------------------ SlimFC head ---------------------------------

class SlimFC(nn.Module):
    in_size: int
    out_size: int
    use_bias: bool = True
    bias_init: float = 0.0
    cust_norm: bool = False
    w_std: float = 0.01  # match normc_initializer(0.01)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(
            features=self.out_size,
            use_bias=self.use_bias,
            bias_init=nn.initializers.constant(self.bias_init),
            kernel_init=normc_initializer(self.w_std),
        )(x)
        if self.cust_norm:
            # Simple norm from the PyTorch version: divide by mean |x| per row,
            # but do not shrink if the mean < 1.
            K = x.shape[-1]
            Gs = jnp.mean(jnp.abs(x), axis=-1, keepdims=True)
            Gs = jnp.maximum(Gs, 1.0)
            x = x / Gs
        return jnp.tanh(x)


# ---------------------------- Softmax Aggregation ----------------------------

class SoftmaxAggregation(nn.Module):
    """Per-feature softmax-weighted sum over the node axis.

    Given X in [B, N, C], computes weights = softmax(X / t, axis=1), and returns
    sum(weights * X, axis=1) in [B, C]. The temperature `t` is learnable.
    """
    init_t: float = 1.0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        t = self.param('t', lambda k: jnp.array(self.init_t, dtype=x.dtype))
        # Stabilize: subtract max over nodes per feature.
        x_scaled = x / (t + 1e-8)
        x_shift = x_scaled - jnp.max(x_scaled, axis=1, keepdims=True)
        w = jax.nn.softmax(x_shift, axis=1)
        return jnp.sum(w * x, axis=1)


# ------------------------------ GATv2 (Star) --------------------------------

class GATv2Star(nn.Module):
    """True GATv2 scoring on a STAR graph per batch item.

    For each graph in the batch: nodes 0..N-1 with self-loops, and edges (j -> 0)
    for all j. We implement the GATv2 attention as in PyTorch Geometric's
    GATv2Conv: for an edge (j -> i),

        e_{i,j,h} = v_h^T LeakyReLU( x_l[i,h,:] + x_r[j,h,:] )
        alpha_{i,j,h} = softmax_j( e_{i,j,h} )
        out_i,h = sum_j alpha_{i,j,h} * x_r[j,h,:]

    where x_l = W_l * x, x_r = W_r * x, both shaped [B, N, H, C]. Only the
    center node i=0 has many incoming edges; non-center nodes aggregate only
    their self-loop, yielding out_j = x_r[j].

    Args:
      in_dim:  input feature dim
      out_dim: per-head output dim (C)
      heads:   number of heads (H)
      negative_slope: LeakyReLU slope
      use_bias: whether W_l/W_r include bias (matches PyG default True)
    """
    in_dim: int
    out_dim: int
    heads: int
    negative_slope: float = 0.2
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [B, N, in_dim]
        B, N, _ = x.shape
        H, C = self.heads, self.out_dim

        # Linear projections for source/target (W_l, W_r) → [B, N, H, C]
        x_l = nn.Dense(features=H * C, use_bias=self.use_bias,
                       kernel_init=nn.initializers.xavier_uniform())(x)
        x_r = nn.Dense(features=H * C, use_bias=self.use_bias,
                       kernel_init=nn.initializers.xavier_uniform())(x)
        x_l = x_l.reshape(B, N, H, C)
        x_r = x_r.reshape(B, N, H, C)

        # Per-head attention vector v_h ~ glorot, shape [H, C]
        v = self.param('att', nn.initializers.xavier_uniform(), (H, C))

        # Center node features (target i = 0) for x_l
        xli_center = x_l[:, :1, :, :]                       # [B, 1, H, C]
        xli_center = jnp.broadcast_to(xli_center, (B, N, H, C))  # [B, N, H, C]

        # Compute logits e_{0,j,h}: v_h^T LeakyReLU(x_l[i=0] + x_r[j])
        e = xli_center + x_r                                 # [B, N, H, C]
        e = jax.nn.leaky_relu(e, negative_slope=self.negative_slope)
        # Dot with v per head
        logits = jnp.einsum('bnhc,hc->bnh', e, v)            # [B, N, H]

        # Softmax over senders j for the center node (i=0)
        alpha = jax.nn.softmax(logits, axis=1)               # [B, N, H]

        # Aggregate messages to the center using x_r (Theta_t * x_j)
        m_center = jnp.sum(alpha[..., None] * x_r, axis=1, keepdims=True)  # [B,1,H,C]

        # Non-center nodes: only self-loop ⇒ output equals x_r[j]
        out = x_r
        out = out.at[:, :1, :, :].set(m_center)              # center replaced by aggregated message

        # Concatenate heads -> [B, N, H*C]
        return out.reshape(B, N, H * C)


# ---------------------------- Encoder & Policy -------------------------------

class GraphEncoder(nn.Module):
    # Mirror key hyperparams from the PyTorch module
    local: bool = True
    num_node_feat: int = 17
    num_node_emb: int = 128
    graph_emb: int = 32
    gat_heads: int = 3
    output_dim: int = 256

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        # Inputs can be [B, N*F] or [B, N, F]
        x = inputs
        if x.ndim == 2:
            # Infer N from last feature size
            F = self.num_node_feat
            x = x.reshape(x.shape[0], -1, F)
        B, N, F = x.shape

        # If local: zero nodes not visible based on last feature flag
        if self.local:
            vis = x[..., -1:]
            x = x * vis

        # Node embed: Dense -> GELU -> LayerNorm
        x = nn.Dense(self.num_node_emb)(x)
        x = gelu(x)
        x = LayerNorm1d()(x)

        # Two GATv2 (star) layers with GELU+LN after each
        x = GATv2Star(in_dim=self.num_node_emb, out_dim=self.graph_emb, heads=self.gat_heads)(x)
        x = gelu(x)
        x = LayerNorm1d()(x)

        x = GATv2Star(in_dim=self.graph_emb * self.gat_heads, out_dim=self.graph_emb, heads=self.gat_heads)(x)
        x = gelu(x)
        x = LayerNorm1d()(x)

        # Global pool via softmax aggregation (per-feature)
        pooled = SoftmaxAggregation(init_t=1.0)(x)  # [B, H*D]

        # FC out: Dense -> GELU -> LayerNorm → [B, output_dim]
        x = nn.Dense(self.output_dim)(pooled)
        x = gelu(x)
        x = LayerNorm1d()(x)
        return x


class GraphPolicy(nn.Module):
    local: bool = True
    num_node_feat: int = 17
    num_node_emb: int = 128
    graph_emb: int = 32
    gat_heads: int = 3
    output_dim: int = 256
    num_actions: int = 4

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        enc = GraphEncoder(
            local=self.local,
            num_node_feat=self.num_node_feat,
            num_node_emb=self.num_node_emb,
            graph_emb=self.graph_emb,
            gat_heads=self.gat_heads,
            output_dim=self.output_dim,
        )(inputs)
        logits = SlimFC(
            in_size=self.output_dim,
            out_size=self.num_actions,
            cust_norm=True,
            w_std=0.01,
        )(enc)
        return logits, enc


# ------------------------------ Example Usage -------------------------------

if __name__ == "__main__":
    key = jax.random.key(0)

    # Dummy batch: B=8, N=16 nodes, F=17 features (last is visibility flag)
    B, N, F = 8, 16, 17
    x = jax.random.normal(key, (B, N, F))

    model = GraphPolicy(local=True, num_node_feat=F, num_actions=4)

    params = model.init(key, x)
    logits, enc = model.apply(params, x)

    print("logits:", logits.shape)  # [8, 4]
    print("enc:", enc.shape)        # [8, 256]

    # JIT compile for performance
    apply_jit = jax.jit(model.apply)
    logits2, enc2 = apply_jit(params, x)
    print("jit ok:", jnp.allclose(logits, logits2))

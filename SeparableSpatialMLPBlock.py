class SeparableSpatialMLPBlock(nn.Module):
    """One block of separable spatial mixing (row -> column) followed by channel mixing.

    Works on inputs shaped [..., H, W, C] (e.g., [B, A, H, W, C]).
    Each mixer is a tiny 2-layer MLP with a residual connection.
    """
    spat_hidden: int
    ch_hidden: int
    activation: str = "relu"
    use_layer_norm: bool = True

    def setup(self) -> None:
        self.activation_fn = _parse_activation_fn(self.activation)

    def _mlp_1d(self, x: chex.Array, hidden: int, out_dim: int) -> chex.Array:
        """A tiny 2-layer MLP applied to the last dimension."""
        if self.use_layer_norm:
            x = nn.LayerNorm(use_scale=False)(x)
        x = nn.Dense(hidden, kernel_init=orthogonal(np.sqrt(2)))(x)
        x = self.activation_fn(x)
        x = nn.Dense(out_dim, kernel_init=orthogonal(1.0))(x)
        return x

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        *lead, H, W, C = x.shape  # accept any leading dims (batch/agents/etc.)
        n = int(np.prod(lead)) if lead else 1
        x_ = x.reshape((n, H, W, C))

        # --- Row mixing: for each (row, channel) mix along width (length W) ---
        y = jnp.transpose(x_, (0, 1, 3, 2))          # (n, H, C, W)
        y = y.reshape((-1, W))                        # (n*H*C, W)
        y = self._mlp_1d(y, self.spat_hidden, W)     # stay at length W
        y = y.reshape((n, H, C, W))
        y = jnp.transpose(y, (0, 1, 3, 2))           # (n, H, W, C)
        x_ = x_ + y                                   # residual

        # --- Column mixing: for each (col, channel) mix along height (length H) ---
        y = jnp.transpose(x_, (0, 2, 3, 1))          # (n, W, C, H)
        y = y.reshape((-1, H))                        # (n*W*C, H)
        y = self._mlp_1d(y, self.spat_hidden, H)     # stay at length H
        y = y.reshape((n, W, C, H))
        y = jnp.transpose(y, (0, 3, 1, 2))           # (n, H, W, C)
        x_ = x_ + y                                   # residual

        # --- Channel mixing: for each spatial site, mix along channels (length C) ---
        y = x_.reshape((-1, C))                       # (n*H*W, C)
        y = self._mlp_1d(y, self.ch_hidden, C)       # stay at C
        y = y.reshape((n, H, W, C))
        x_ = x_ + y                                   # residual

        return x_.reshape((*lead, H, W, C))

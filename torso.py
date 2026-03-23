class MLPTorsoSetTxMinimal(nn.Module):
    """
    Minimal incremental torso:
      - keep old MLP-style policy interface
      - keep old head unchanged
      - insert ONE SetTx block only in the neighbor token path

    Assumed obs layout (matching your updated env):
      [ ego | slots*per_slot_d | pairwise_dists | terrain ]
    with:
      slots = 5
      per_slot_d = 15
      pairwise_dists = 10
      terrain = 9x9 = 81
    """

    slots: int = 5
    per_slot_d: int = 15
    view: int = 9

    nbr_token_dim: int = 24
    hidden_dim: int = 128

    settx_heads: int = 4
    settx_mlp_hidden: int = 64

    ego_hidden: int = 64
    pair_hidden: int = 32
    ter_hidden: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.float32)
        B, N, D = x.shape

        pair_d = self.slots * (self.slots - 1) // 2
        ter_d = self.view * self.view
        nbr_d = self.slots * self.per_slot_d + pair_d
        ego_d = D - nbr_d - ter_d
        if ego_d <= 0:
            raise ValueError(
                f"Inferred ego_d={ego_d} not positive. "
                f"Check layout: D={D}, nbr_d={nbr_d}, ter_d={ter_d}"
            )

        ego = x[..., :ego_d]                                  # (B,N,ego_d)
        nbr_all = x[..., ego_d : ego_d + nbr_d]              # (B,N,nbr_d)
        ter = x[..., -ter_d:]                                # (B,N,81)

        nbr_main = nbr_all[..., : self.slots * self.per_slot_d].reshape(
            B, N, self.slots, self.per_slot_d
        )                                                    # (B,N,K,15)
        pair = nbr_all[..., self.slots * self.per_slot_d :]  # (B,N,10)

        # ------------------------------------------
        # Neighbor path: this is the only real change
        # ------------------------------------------

        # Per-slot split; only r is needed for mask.
        r = nbr_main[..., 0:3]                               # (B,N,K,3)
        mask = jnp.any(jnp.abs(r) > 1e-6, axis=-1)           # (B,N,K) bool

        # Old-style per-slot embedding
        nbr_tok = nn.Dense(
            self.nbr_token_dim,
            kernel_init=orthogonal(jnp.sqrt(2.0)),
        )(nbr_main)
        nbr_tok = nn.gelu(nbr_tok)
        nbr_tok = zero_if_masked(nbr_tok, mask)

        # One tiny SetTx block over the 5 slots
        flat_tok = nbr_tok.reshape(B * N, self.slots, self.nbr_token_dim)
        flat_mask = mask.reshape(B * N, self.slots)
        flat_tok = NeighborSetTxBlock(
            d_model=self.nbr_token_dim,
            num_heads=self.settx_heads,
            mlp_hidden=self.settx_mlp_hidden,
        )(flat_tok, flat_mask)
        nbr_tok = flat_tok.reshape(B, N, self.slots, self.nbr_token_dim)

        # Keep old-style summaries after contextualization
        nbr_c = nbr_tok.reshape(B, N, self.slots * self.nbr_token_dim)  # slot-order-sensitive path

        nbr_sum = jnp.sum(zero_if_masked(nbr_tok, mask), axis=-2)
        nbr_cnt = jnp.maximum(jnp.sum(mask, axis=-1, keepdims=True), 1)
        nbr_mean = nbr_sum / nbr_cnt

        neg_big = jnp.full_like(nbr_tok, -1e9)
        nbr_max = jnp.max(jnp.where(mask[..., None], nbr_tok, neg_big), axis=-2)
        nbr_any = jnp.any(mask, axis=-1, keepdims=True)
        nbr_max = jnp.where(nbr_any, nbr_max, 0.0)

        # ------------------------------------------
        # Keep the rest old/simple
        # ------------------------------------------

        ego_e = nn.Dense(self.ego_hidden, kernel_init=orthogonal(jnp.sqrt(2.0)))(ego)
        ego_e = nn.gelu(ego_e)

        pair_e = nn.Dense(self.pair_hidden, kernel_init=orthogonal(jnp.sqrt(2.0)))(pair)
        pair_e = nn.gelu(pair_e)

        ter_e = nn.Dense(self.ter_hidden, kernel_init=orthogonal(jnp.sqrt(2.0)))(ter)
        ter_e = nn.gelu(ter_e)

        z = jnp.concatenate([ego_e, pair_e, ter_e, nbr_c, nbr_mean, nbr_max], axis=-1)

        z = nn.Dense(self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2.0)))(z)
        z = nn.gelu(z)
        z = nn.LayerNorm()(z)

        z = nn.Dense(self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2.0)))(z)
        z = nn.gelu(z)
        z = nn.LayerNorm()(z)

        return z

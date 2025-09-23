        B, A, K, F_n = entities.shape
        _, _, F_e = ego.shape

        # Broadcast ego to neighbour positions and concat with neighbour features.
        ego_b = jnp.broadcast_to(ego[:, :, None, :], (B, A, K, F_e))  # (B, A, K, F_e)
        neigh_cat = jnp.concatenate([entities, ego_b], axis=-1)       # (B, A, K, F_n+F_e)

        # Shared MLP over the feature dimension at each neighbour position (1x1 conv equivalent).
        phi = _mlp(self.message_hidden, out_size=self.message_dim, act=nn.swish)
        msgs = phi(neigh_cat)  # (B, A, K, message_dim)

        # Attention logits per neighbour (conditioned on ego as well).
        psi = _mlp(self.attn_hidden, out_size=1, act=nn.swish)
        attn_logits = psi(neigh_cat)[..., 0]  # (B, A, K)

        # Normalize across neighbours; mask can be added if some slots are padding.
        attn = nn.softmax(attn_logits, axis=-1)                      # (B, A, K)
        attn = nn.Dropout(self.dropout_rate, deterministic=not train)(attn[..., None])  # (B, A, K, 1)

        # Weighted sum over neighbours.
        M = jnp.sum(attn * msgs, axis=2)

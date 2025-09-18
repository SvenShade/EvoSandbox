class MLP(nn.Module):
    hidden: int
    out: int
    act: Callable = gelu
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden, use_bias=True)(x)
        x = self.act(x)
        x = nn.Dense(self.out, use_bias=True)(x)
        return x

class MixerBlock(nn.Module):
    token_mlp_dim: int
    channel_mlp_dim: int
    act: Callable = gelu
    @nn.compact
    def __call__(self, x):
        # x: (T, C) or (B, T, C)
        has_batch = (x.ndim == 3)
        if not has_batch:
            x = x[None, ...]  # -> (1, T, C)

        # Token mixing (along T)
        y = nn.LayerNorm()(x)
        y = jnp.swapaxes(y, 1, 2)                       # (B, C, T)
        y = MLP(self.token_mlp_dim, y.shape[-1], self.act)(y)
        y = jnp.swapaxes(y, 1, 2)                       # (B, T, C)
        x = x + y

        # Channel mixing (along C)
        y = nn.LayerNorm()(x)
        y = MLP(self.channel_mlp_dim, y.shape[-1], self.act)(y)
        x = x + y

        return x if has_batch else x[0]

class MLPMixerEncoder(nn.Module):
    num_layers: int
    token_mlp_dim: int
    channel_mlp_dim: int
    embed_dim: int
    act: Callable = gelu
    in_proj_dim: Optional[int] = None  # optional 1x linear to widen channel dim

    @nn.compact
    def __call__(self, x_seq):
        # x_seq: (T, C) or (B, T, C)
        if self.in_proj_dim is not None:
            x_seq = nn.Dense(self.in_proj_dim)(x_seq)

        for _ in range(self.num_layers):
            x_seq = MixerBlock(self.token_mlp_dim, self.channel_mlp_dim, act=self.act)(x_seq)

        x_seq = nn.LayerNorm()(x_seq)
        pooled = jnp.mean(x_seq, axis=1) if x_seq.ndim == 3 else jnp.mean(x_seq, axis=0)
        return nn.Dense(self.embed_dim)(pooled)

# ----------------- patchify helper -----------------

def patchify_hw(x, patch: Tuple[int, int]):
    """
    x: (H, W) or (H, W, C)
    Returns: (T, Cp) where T = (H/ph)*(W/pw), Cp = ph*pw*C
    """
    assert x.ndim in (2, 3), "terrain must be (H, W) or (H, W, C)"
    if x.ndim == 2:
        H, W = x.shape
        C = 1
        x = x[..., None]  # -> (H, W, 1)
    else:
        H, W, C = x.shape

    ph, pw = patch
    assert (H % ph == 0) and (W % pw == 0), "H and W must be divisible by patch size"

    # Reshape to (H/ph, ph, W/pw, pw, C)
    x = x.reshape(H // ph, ph, W // pw, pw, C)
    # Move patch dims together and grid dims together: (H/ph, W/pw, ph*pw*C)
    x = jnp.transpose(x, (0, 2, 1, 3, 4))  # (H/ph, W/pw, ph, pw, C)
    x = x.reshape((H // ph) * (W // pw), ph * pw * C)  # (T, Cp)
    return x  # tokens

# ----------------- policy torso with patchified terrain -----------------

class FastMixerPolicyTorso(nn.Module):
    # Entity mixer
    ent_layers: int = 2
    ent_token_mlp_dim: int = 64
    ent_channel_mlp_dim: int = 128

    # Terrain mixer + patching
    ter_layers: int = 2
    ter_token_mlp_dim: int = 64
    ter_channel_mlp_dim: int = 128
    ter_patch: Tuple[int, int] = (3, 3)  # e.g., 3x3 patches
    ter_in_proj_dim: int = 32            # project Cp -> this before mixing

    # Self MLP
    self_hidden: int = 128

    # Fused embedding size
    embed_dim: int = 256
    act: Callable = gelu

    @nn.compact
    def __call__(self, entities, terrain, self_feats):
        # --- Entities: (N, Fe) ---
        assert entities.ndim == 2, "entities must be (N, Fe)"
        ent_embed = MLPMixerEncoder(
            num_layers=self.ent_layers,
            token_mlp_dim=self.ent_token_mlp_dim,
            channel_mlp_dim=self.ent_channel_mlp_dim,
            embed_dim=self.embed_dim,
            act=self.act,
            in_proj_dim=None,
        )(entities)

        # --- Terrain: patchify (H, W) or (H, W, C) -> (T, Cp) ---
        tokens = patchify_hw(terrain, self.ter_patch)  # (T, Cp)
        ter_embed = MLPMixerEncoder(
            num_layers=self.ter_layers,
            token_mlp_dim=self.ter_token_mlp_dim,
            channel_mlp_dim=self.ter_channel_mlp_dim,
            embed_dim=self.embed_dim,
            act=self.act,
            in_proj_dim=self.ter_in_proj_dim,  # cheap lift to channel width
        )(tokens)

        # --- Self: (Fs,) ---
        assert self_feats.ndim == 1, "self must be (Fs,)"
        self_embed = MLP(self_hidden=self.self_hidden, out=self.embed_dim, act=self.act)(self_feats)

        # --- Fuse ---
        fused = jnp.concatenate([ent_embed, ter_embed, self_embed], axis=-1)  # (3*embed_dim,)
        fused = nn.LayerNorm()(fused)
        fused = nn.Dense(self.embed_dim)(fused)
        return fused  # (embed_dim,)

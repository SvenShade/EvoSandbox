
class AntisymMatcher(nn.Module):
    def __init__(
        self,
        n_vertices: int,
        in_channels: int,
        rank: int,
        heads: int,
        dropout: float = 0.1,
        topk_k: int | None = 12,
    ):
        super().__init__()
        self.n_vertices = n_vertices
        self.in_channels = in_channels
        self.rank = rank
        self.heads = heads
        self.topk_k = topk_k

        # Projections for multi-head low-rank bilinear compatibility
        self.q_proj = nn.Linear(in_channels, rank * heads, bias=False)
        self.k_proj = nn.Linear(in_channels, rank * heads, bias=False)

        self.do = nn.Dropout(dropout)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.head_scale = 1.0 / math.sqrt(rank)

        # Diagonal bias: one scalar per vertex computed from its feature
        self.diag_proj = nn.Linear(in_channels, 1, bias=True)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        feats = feats[:, 1:]
        B, L, C = feats.shape
        feats = feats.view(B, L // 2, 2, C).mean(dim=2)
        N = feats.size(1)

        x = F.layer_norm(feats, (C,))
        x = self.do(x)
        Q = self.q_proj(x).view(B, N, self.heads, self.rank)
        K = self.k_proj(x).view(B, N, self.heads, self.rank)
        A = torch.einsum('b n h r, b m h r -> b h n m', Q, K) * self.head_scale
        A = A.sum(dim=1)  # [B, N, N]

        if self.topk_k is not None and self.topk_k > 0 and self.topk_k < N:
            base = A
            k = self.topk_k
            row_idx = base.topk(k, dim=-1).indices
            col_idx = base.topk(k, dim=-2).indices
            mask = torch.zeros_like(base, dtype=torch.bool)
            mask.scatter_(-1, row_idx, True)
            mask.scatter_(-2, col_idx, True)
            # Bias against unlikely long-range pairs
            A = A.masked_fill(~mask, A.new_full((), -10.0))

        S = 0.5 * (A - A.transpose(1, 2))
        diag_bias = self.diag_proj(x).squeeze(-1)
        S = S + torch.diag_embed(diag_bias)
        return S * self.logit_scale.clamp_min(1e-2)


class EncoderDecoder(nn.Module):
    """
    Drop-in replacement that swaps the two ScoreNet heads for AntisymMatcher.
    - forward(image, tgt)  -> (preds_f, perm_mat)  # unchanged
    - predict(image, tgt)  -> (preds, feats)       # unchanged

    Assumes `log_optimal_transport(scores, bin_score, iters)` is available in scope.
    """
    def __init__(self, cfg, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.decoder = decoder

        self.matcher = AntisymMatcher(
            n_vertices=getattr(cfg, 'N_VERTICES', 200),
            in_channels=256,  # match your decoder token width
            rank=getattr(cfg, 'MATCHER_RANK', 32),
            heads=getattr(cfg, 'MATCHER_HEADS', 6),
            topk_k=getattr(cfg, 'MATCHER_TOPK', 12),
        )

        # Preserve the same learnable bin score parameter
        self.bin_score = nn.Parameter(torch.tensor(1.0))

    def forward(self, image: torch.Tensor, tgt: torch.Tensor):
        """
        Training path:
          - returns (preds_f, perm_mat) where perm_mat is soft row-probabilities
            (as in the original: log_optimal_transport -> row softmax)
        """
        encoder_out = self.encoder(image)
        preds_f, feats = self.decoder(encoder_out, tgt)

        logits = self.matcher(feats)  # [B, N, N]
        perm_mat = log_optimal_transport(logits, self.bin_score, self.cfg.SINKHORN_ITERATIONS)
        perm_mat = F.softmax(perm_mat, dim=-1)  # BCE expects probabilities

        return preds_f, perm_mat

    @torch.no_grad()
    def predict(self, image: torch.Tensor, tgt: torch.Tensor):
        """
        Inference path:
          - returns (preds, feats) for external Hungarian, identical to original.
        """
        encoder_out = self.encoder(image)
        preds, feats = self.decoder.predict(encoder_out, tgt)
        return preds, feats

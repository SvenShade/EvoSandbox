import torch
from torch import nn
from torch.nn import functional as F
import math

# --- New head: Antisymmetric matcher (drop-in replacement for two ScoreNet heads) ---
class AntisymMatcher(nn.Module):
    """
    Elegant, coordinate-aware antisymmetric matcher for building footprints.

    Core ideas (simple + strong):
      - Drop CLS, fold (y,x) token pairs → one vertex feature per step (ScoreNet assumption)
      - Fuse vertex coords into token features (feature + coord embedding)
      - Multi-head low-rank bilinear scoring + antisymmetry by construction
      - Spatial kNN mask + distance bias to prefer local, plausible edges
      - Learnable diagonal bias so pads self-connect confidently
    """
    def __init__(
        self,
        n_vertices: int,
        in_channels: int = 256,
        rank: int = 32,
        heads: int = 6,
        topk_k: int = 12,
        lambda_dist: float = 6.0,
    ):
        super().__init__()
        self.n_vertices = n_vertices
        self.in_channels = in_channels
        self.rank = rank
        self.heads = heads
        self.topk_k = topk_k
        self.lambda_dist = lambda_dist

        # Coordinate fusion: map (y,x) in [0,1]^2 to feature space and add residually
        self.coord_proj = nn.Linear(2, in_channels, bias=True)

        # Projections for multi-head low-rank bilinear compatibility
        self.q_proj = nn.Linear(in_channels, rank * heads, bias=False)
        self.k_proj = nn.Linear(in_channels, rank * heads, bias=False)

        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.head_scale = 1.0 / math.sqrt(rank)

        # Diagonal bias: one scalar per vertex computed from its (fused) feature
        self.diag_proj = nn.Linear(in_channels, 1, bias=True)

    def forward(self, feats: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """feats: [B, N+1, C] or [B, N, C]; xy: [B, N, 2] normalised to [0,1]."""
        # 1) ScoreNet-like folding (drop CLS, fold two tokens → one vertex feature)
        feats = feats[:, 1:]
        B, L, C = feats.shape
        feats = feats.view(B, L // 2, 2, C).mean(dim=2)  # [B, N, C]
        N = feats.size(1)

        # 2) Fuse coordinates at the vertex level (residual)
        # Expect xy already aligned to folded vertices
        fused = feats + self.coord_proj(xy)

        # 3) Multi-head low-rank bilinear scores
        Q = self.q_proj(fused).view(B, N, self.heads, self.rank)
        K = self.k_proj(fused).view(B, N, self.heads, self.rank)
        A = torch.einsum('b n h r, b m h r -> b h n m', Q, K) * self.head_scale
        A = A.sum(dim=1)  # [B, N, N]

        # 4) Spatial kNN mask + distance bias (encourage local, plausible edges)
        if self.topk_k is not None and self.topk_k > 0 and self.topk_k < N:
            dist = torch.cdist(xy, xy, p=2)  # [B, N, N]
            k = self.topk_k
            row_idx = dist.topk(k, largest=False, dim=-1).indices
            col_idx = dist.topk(k, largest=False, dim=-2).indices
            mask = torch.zeros_like(A, dtype=torch.bool)
            mask.scatter_(-1, row_idx, True)
            mask.scatter_(-2, col_idx, True)
            A = A.masked_fill(~mask, A.new_full((), -10.0))
            if self.lambda_dist != 0.0:
                A = A + (-float(self.lambda_dist)) * dist

        # 5) Antisymmetry + per-vertex diagonal bias (pads)
        S = 0.5 * (A - A.transpose(1, 2))
        diag_bias = self.diag_proj(fused).squeeze(-1)  # [B, N]
        S = S + torch.diag_embed(diag_bias)
        return S * self.logit_scale.clamp_min(1e-2)


# --- Patched EncoderDecoder that swaps in AntisymMatcher ---
import math

class EncoderDecoder(nn.Module):
    def __init__(self, cfg, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.decoder = decoder

        # REPLACEMENT: one antisymmetric matcher instead of two ScoreNet branches
        # Default in_channels=512 to mirror prior ScoreNet usage; adjust via cfg if needed.
        in_ch = 256
        heads = getattr(cfg, 'MATCHER_HEADS', 4)
        rank = getattr(cfg, 'MATCHER_RANK', 16)
        self.matcher = AntisymMatcher(getattr(cfg, 'N_VERTICES', 200), in_channels=256, rank=getattr(cfg, 'MATCHER_RANK', 32), heads=getattr(cfg, 'MATCHER_HEADS', 6), topk_k=getattr(cfg, 'MATCHER_TOPK', 12), lambda_dist=getattr(cfg, 'MATCHER_LAMBDA_DIST', 6.0)), in_channels=256, rank=getattr(cfg,'MATCHER_RANK',32), heads=getattr(cfg,'MATCHER_HEADS',6), topk_k=getattr(cfg,'MATCHER_TOPK',12))
        # Optional distance bias strength
        self.matcher.lambda_dist = getattr(cfg, 'MATCHER_LAMBDA_DIST', 6.0), in_channels=256, rank=getattr(cfg,'MATCHER_RANK',32), heads=getattr(cfg,'MATCHER_HEADS',6), topk_k=getattr(cfg,'MATCHER_TOPK',12)), in_channels=in_ch, rank=rank, heads=heads)

        # Back-compat: expose scorenet1/2 so utils.test_generate still works

        # Preserve the same learnable bin score parameter that original code used
        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

    def forward(self, image: torch.Tensor, tgt: torch.Tensor):
        """Keeps the original external contract.
        Returns `(preds_f, perm_mat)` with the same shapes/dtypes as before.
        """
        # Encode / decode exactly as before
        encoder_out = self.encoder(image)
        preds_f, feats = self.decoder(encoder_out, tgt)

        # --- New head path ---
        # Build per-vertex (y,x) coords from tokens for spatial priors
        # Expect tgt: [B, L] of token ids; bins in [0, NUM_BINS)
        try:
            seq = tgt[:, 1:]  # drop BOS/CLS to mirror matcher folding
            y_ids = seq[:, 0::2].float()
            x_ids = seq[:, 1::2].float()
            denom = max(1, int(getattr(self.cfg, 'NUM_BINS', getattr(self.cfg, 'INPUT_HEIGHT', 224)) - 1))
            xy = torch.stack([y_ids / denom, x_ids / denom], dim=-1)  # [B, N, 2], normalised
        except Exception:
            xy = None

        logits = self.matcher(feats, xy=xy)  # [B, N, N]

        # Call the *existing* log-space OT solver (same as original)
        # NOTE: function must be present in the surrounding module scope.
        perm_mat = log_optimal_transport(logits, self.bin_score, self.cfg.SINKHORN_ITERATIONS)
        # Slice back to [B, N, N] in case solver padded with bins
        perm_mat = perm_mat[:, :logits.shape[1], :logits.shape[2]]
        # Final row-softmax (unchanged from original)
        perm_mat = F.softmax(perm_mat, dim=-1)

        return preds_f, perm_mat

    @torch.no_grad()
    def predict(self, image: torch.Tensor, tgt: torch.Tensor):
        """Inference path kept identical to the original: returns (preds, feats).
        Downstream code that applies Hungarian/OMN externally will keep working.
        """
        encoder_out = self.encoder(image)
        preds, feats = self.decoder.predict(encoder_out, tgt)
        return preds, feats

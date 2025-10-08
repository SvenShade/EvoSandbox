import torch
from torch import nn
from torch.nn import functional as F
import math

# --- New head: Antisymmetric matcher (drop-in replacement for two ScoreNet heads) ---
class AntisymMatcher(nn.Module):
    """
    Multi-head antisymmetric matcher for building footprints.

    Identical assumptions to ScoreNet:
      - drop CLS
      - fold every 2 tokens → 1 vertex feature by mean
      - no diagonal suppression; pads handled by OT dustbin
      - no padding/truncation inside the head
    """
    def __init__(self, n_vertices: int, in_channels: int = 256, rank: int = 16, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_vertices = n_vertices
        self.in_channels = in_channels
        self.rank = rank
        self.heads = heads

        self.q_proj = nn.Linear(in_channels, rank * heads, bias=False)
        self.k_proj = nn.Linear(in_channels, rank * heads, bias=False)
        self.do = nn.Dropout(dropout)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.head_scale = 1.0 / math.sqrt(rank)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        if feats.dim() != 3:
            raise ValueError(f"AntisymMatcher expects [B, N, C] or [B, N+1, C], got {feats.shape}")
        # Mirror ScoreNet: drop CLS, fold pairs (2 tokens per vertex)
        feats = feats[:, 1:]
        B, L, C = feats.shape
        feats = feats.view(B, L // 2, 2, C).mean(dim=2)   # [B, N, C], N = L//2
        N = feats.size(1)

        # Lightweight normalization and projections
        x = F.layer_norm(feats, (C,))
        x = self.do(x)
        Q = self.q_proj(x).view(B, N, self.heads, self.rank)
        K = self.k_proj(x).view(B, N, self.heads, self.rank)

        # Multi-head low-rank bilinear scores
        A = torch.einsum('b n h r, b m h r -> b h n m', Q, K) * self.head_scale
        A = A.sum(dim=1)  # [B, N, N]

        # Antisymmetry (no diagonal masking)
        S = 0.5 * (A - A.transpose(1, 2))
        return S * self.logit_scale.clamp_min(1e-2)


# --- Patched EncoderDecoder that swaps in AntisymMatcher ---
import math
class _MatcherShim(nn.Module):
    """Shim to preserve model.scorenet1/2 calls used in utils.test_generate.
    scorenet1(feats) -> matcher(feats); scorenet2(feats) -> zeros.
    """
    def __init__(self, matcher: nn.Module, n_vertices: int, mode: str):
        super().__init__()
        assert mode in {"pass", "zeros"}
        self.matcher = matcher
        self.n_vertices = n_vertices
        self.mode = mode
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        if self.mode == "pass":
            return self.matcher(feats)
        # zeros path: produce [B, N_cfg, N_cfg] zeros on the correct device/dtype
        B = feats.size(0)
        N = self.n_vertices
        return feats.new_zeros((B, N, N))


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
        self.matcher = AntisymMatcher(getattr(cfg, 'N_VERTICES', 200), in_channels=in_ch, rank=rank, heads=heads)

        # Back-compat: expose scorenet1/2 so utils.test_generate still works
        self.scorenet1 = _MatcherShim(self.matcher, n_vertices=self.cfg.N_VERTICES, mode="pass")
        self.scorenet2 = _MatcherShim(self.matcher, n_vertices=self.cfg.N_VERTICES, mode="zeros")

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
        logits = self.matcher(feats)  # [B, N, N]

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

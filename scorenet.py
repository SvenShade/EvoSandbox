import math
import torch
from torch import nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------
# ScoreNet (Minimal, Efficient Drop-in)
# -----------------------------------------------------------------------------
# API & assumptions preserved:
#   • Input feats: [B, N(+1), C] from decoder (with CLS/BOS at index 0)
#   • Behavior: drop CLS, fold (y,x) token pairs → vertices [B, N, C]
#   • Output: raw unconstrained scores [B, N, N] (no antisymmetry, no diag mask)
#
# Implementation:
#   Replace the N×N 1×1-conv stack with a low-rank, multi-head bilinear form
#   computed via GEMMs (einsum). Add a small per-vertex diagonal bias so pads
#   can self-loop strongly.
# -----------------------------------------------------------------------------

class ScoreNet(nn.Module):
    def __init__(self, in_channels: int = 256, heads: int = 6, rank: int = 32):
        super().__init__()
        self.C, self.H, self.R = in_channels, heads, rank

        # Per-head low-rank projections
        self.q_proj = nn.Linear(in_channels, heads * rank, bias=False)
        self.k_proj = nn.Linear(in_channels, heads * rank, bias=False)
        self.head_scale = 1.0 / math.sqrt(rank)

        # Per-vertex diagonal bias (pads/self-loops)
        self.diag_proj = nn.Linear(in_channels, 1, bias=True)

        # Optional global temperature for logits
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _fold_vertices(feats: torch.Tensor) -> torch.Tensor:
        # Drop CLS, fold two tokens (y,x) → one vertex feature by mean
        feats = feats[:, 1:]
        B, L, C = feats.shape
        return feats.view(B, L // 2, 2, C).mean(dim=2)  # [B, N, C]

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        feats: [B, N+1, C] or [B, N, C]  ->  scores S: [B, N, N]
        """
        x = self._fold_vertices(feats)                  # [B, N, C]
        B, N, C = x.shape

        # Multi-head low-rank bilinear scores
        Q = self.q_proj(x).view(B, N, self.H, self.R)   # [B, N, H, R]
        K = self.k_proj(x).view(B, N, self.H, self.R)   # [B, N, H, R]
        S = torch.einsum('b n h r, b m h r -> b h n m', Q, K) * self.head_scale  # [B, H, N, N]
        S = S.sum(dim=1)                                # [B, N, N]

        # Diagonal bias for pads
        S = S + torch.diag_embed(self.diag_proj(x).squeeze(-1))

        return S * self.logit_scale.clamp_min(1e-2)


# -----------------------------------------------------------------------------
# EncoderDecoder (Minimal two-head OMN wrapper; optional)
# -----------------------------------------------------------------------------
# Mirrors original OMN: two heads combined as P1 + P2ᵀ, then Sinkhorn/OT.
# Use this wrapper only if you want to swap at the entry module level;
# otherwise instantiate ScoreNet twice where your original code did.
# -----------------------------------------------------------------------------

class EncoderDecoder(nn.Module):
    """
    Assumes `log_optimal_transport(scores, bin_score, iters)` is defined in scope.
    - forward(image, tgt)  -> (preds_f, perm_mat)
    - predict(image, tgt)  -> (preds, feats)
    """
    def __init__(self, cfg, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.cfg, self.encoder, self.decoder = cfg, encoder, decoder
        heads = getattr(cfg, 'SCORENET_HEADS', 6)
        rank  = getattr(cfg, 'SCORENET_RANK', 32)
        C     = getattr(cfg, 'IN_CHANNELS', 256)  # or fix to 256 if preferred

        self.scorenet1 = ScoreNet(in_channels=C, heads=heads, rank=rank)
        self.scorenet2 = ScoreNet(in_channels=C, heads=heads, rank=rank)

        # Same learnable bin score as original
        self.bin_score = nn.Parameter(torch.tensor(1.0))

    def forward(self, image: torch.Tensor, tgt: torch.Tensor):
        enc = self.encoder(image)
        preds_f, feats = self.decoder(enc, tgt)

        P1 = self.scorenet1(feats)               # [B, N, N]
        P2 = self.scorenet2(feats).transpose(1, 2)
        logits = P1 + P2                          # combine like original

        perm = log_optimal_transport(logits, self.bin_score, self.cfg.SINKHORN_ITERATIONS)
        perm = F.softmax(perm, dim=-1)            # BCE expects probabilities
        return preds_f, perm

    @torch.no_grad()
    def predict(self, image: torch.Tensor, tgt: torch.Tensor):
        enc = self.encoder(image)
        preds, feats = self.decoder.predict(enc, tgt)
        return preds, feats

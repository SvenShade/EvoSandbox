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



import math
import torch
from torch import nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------
# ScoreNet (Minimal + Pair Nonlinearity)
# -----------------------------------------------------------------------------
# Preserves the original API/assumptions:
#   • Input feats: [B, N(+1), C]  (decoder hidden states; first token is CLS/BOS)
#   • Behavior: drop CLS, fold (y,x) pairs → [B, N, C]
#   • Output: raw scores [B, N, N] (no antisymmetry, no diag mask)
#
# Adds:
#   • Tiny separable MLP on vertices (f_i -> a_i, f_j -> b_j), then outer-product
#     S_pair(i,j) = <a_i, b_j> to recover some of the conv-head’s gating/logic.
#   • Weighted sum: S = S_bilinear + alpha_pair * S_pair + diag_bias
# -----------------------------------------------------------------------------

class ScoreNet(nn.Module):
    def __init__(self, in_channels: int = 256, heads: int = 6, rank: int = 32,
                 pair_hidden: int = 16, alpha_pair: float = 0.5):
        super().__init__()
        self.C, self.H, self.R = in_channels, heads, rank

        # Bilinear core (fast)
        self.q_proj = nn.Linear(in_channels, heads * rank, bias=False)
        self.k_proj = nn.Linear(in_channels, heads * rank, bias=False)
        self.head_scale = 1.0 / math.sqrt(rank)

        # NEW: tiny separable MLP for pair nonlinearity
        self.ph_i = nn.Linear(in_channels, pair_hidden, bias=True)  # f_i -> a_i
        self.ph_j = nn.Linear(in_channels, pair_hidden, bias=True)  # f_j -> b_j
        self.alpha_pair = nn.Parameter(torch.tensor(alpha_pair, dtype=torch.float32))

        # Diagonal bias (pads/self-loops)
        self.diag_proj = nn.Linear(in_channels, 1, bias=True)

        # Global temperature for logits
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _fold_vertices(feats: torch.Tensor) -> torch.Tensor:
        # Drop CLS/BOS, fold two tokens (y,x) → one vertex feature by mean
        feats = feats[:, 1:]
        B, L, C = feats.shape
        return feats.view(B, L // 2, 2, C).mean(dim=2)  # [B, N, C]

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = self._fold_vertices(feats)                    # [B, N, C]
        B, N, C = x.shape

        # --- Bilinear scores (as before) ---
        Q = self.q_proj(x).view(B, N, self.H, self.R)     # [B,N,H,R]
        K = self.k_proj(x).view(B, N, self.H, self.R)     # [B,N,H,R]
        S = torch.einsum('b n h r, b m h r -> b h n m', Q, K) * self.head_scale
        S = S.sum(dim=1)                                   # [B, N, N]

        # --- NEW: Pair nonlinearity (separable MLP + outer-product) ---
        a_i = F.relu(self.ph_i(x))                        # [B, N, Dp]
        b_j = F.relu(self.ph_j(x))                        # [B, N, Dp]
        S_pair = torch.einsum('b n d, b m d -> b n m', a_i, b_j)  # [B, N, N]
        S = S + self.alpha_pair.clamp_min(0.0) * S_pair

        # Diagonal bias (pads)
        S = S + torch.diag_embed(self.diag_proj(x).squeeze(-1))

        return S * self.logit_scale.clamp_min(1e-2)


# Centering in EncoderDecoder.forward:
logits = perm_mat1 + perm_mat2.transpose(1, 2)  # [B,N,N]

# BN2d-ish centering (cheap, stateless)
logits = logits - logits.mean(dim=-1, keepdim=True)  # zero-mean rows
logits = logits - logits.mean(dim=-2, keepdim=True)  # zero-mean cols

perm_mat = log_optimal_transport(logits, self.bin_score, self.cfg.SINKHORN_ITERATIONS)
perm_mat = F.softmax(perm_mat, dim=-1)



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

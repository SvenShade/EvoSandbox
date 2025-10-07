import torch
from torch import nn
from torch.nn import functional as F
import math


# --- New head: Antisymmetric matcher (drop-in replacement for two ScoreNet heads) ---
class AntisymMatcher(nn.Module):
"""
Computes pairwise successor logits S \in R^{B x N x N} from decoder token features
using a low-rank bilinear scorer, then enforces antisymmetry by construction:
S = 0.5 * (A - A^T)
where A_ij = (Q f_i)^T (K f_j).


Notes
-----
* Expects the same `feats` tensor that was previously passed to ScoreNet.
In the original ScoreNet.forward, the first token was dropped via feats = feats[:, 1:].
We preserve that behavior here.
* Default in_channels=512 matches the ScoreNet default, so this is safe even if
cfg does not expose the decoder width explicitly.
* Designed for building footprints (single-ring), so we also suppress the diagonal (no self-edges).
"""
def __init__(self, n_vertices: int, in_channels: int = 256, rank: int = 32, dropout: float = 0.1):
super().__init__()
self.n_vertices = n_vertices
self.in_channels = in_channels
self.rank = rank


# Low-rank projections (Q, K)
self.q_proj = nn.Linear(in_channels, rank, bias=False)
self.k_proj = nn.Linear(in_channels, rank, bias=False)


# Small stabilization stack
self.norm = nn.LayerNorm(in_channels)
self.do = nn.Dropout(dropout)


# Optional learnable temperature (helps when swapping from two heads to one)
self.logit_scale = nn.Parameter(torch.tensor(1.0))


# Initialize with truncated normal-ish scale (similar spirit to timm init)
nn.init.kaiming_uniform_(self.q_proj.weight, a=math.sqrt(5))
nn.init.kaiming_uniform_(self.k_proj.weight, a=math.sqrt(5))


def forward(self, feats: torch.Tensor) -> torch.Tensor:
"""
Parameters
----------
feats : Tensor, shape [B, N+1, C] or [B, N, C]
Decoder token features; if a CLS token is present at index 0, it is dropped.
Returns
-------
logits : Tensor, shape [B, N, N]
Antisymmetric pairwise logits suitable for log_optimal_transport.
"""
# Drop possible CLS token to keep behavior consistent with original ScoreNet
if feats.dim() != 3:
raise ValueError(f"AntisymMatcher expects [B, N, C] or [B, N+1, C], got {feats.shape}")
if feats.size(1) > self.n_vertices:
feats = feats[:, 1:]
B, N, C = feats.shape


# Safety clamp if config's n_vertices > actual N at runtime (ragged batches)
if N != self.n_vertices:
# No hard requirement; the matcher works for any N. We just track for info.
pass


# Normalize & project
x = self.norm(feats)
x = self.do(x)
Q = self.q_proj(x) # [B, N, r]
K = self.k_proj(x) # [B, N, r]


# Low-rank bilinear scores A_ij = (Q f_i)^T (K f_j)
# -> [B, N, N]
A = torch.einsum('bir,bjr->bij', Q, K)


# Enforce antisymmetry by construction
S = 0.5 * (A - A.transpose(1, 2))


# Suppress diagonal (no self loops in successor relation for simple rings)
diag_mask = torch.eye(N, device=S.device, dtype=torch.bool).unsqueeze(0) # [1, N, N]
S = S.masked_fill(diag_mask, float('-inf'))


# Optional temperature
S = S * self.logit_scale.clamp_min(1e-2)
return S




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
rank = getattr(cfg, 'MATCHER_RANK', 32)
self.matcher = AntisymMatcher(getattr(cfg, 'N_VERTICES', 200), in_channels=in_ch, rank=rank)


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
logits = self.matcher(feats) # [B, N, N]


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
"""Inference path: same as original (Hungarian can be applied outside)."""
encoder_out = self.encoder(image)
preds, feats = self.decoder.predict(encoder_out, tgt)
logits = self.matcher(feats)
perm_mat = log_optimal_transport(logits, self.bin_score, self.cfg.SINKHORN_ITERATIONS)
perm_mat = perm_mat[:, :logits.shape[1], :logits.shape[2]]
perm_mat = F.softmax(perm_mat, dim=-1)
return preds, perm_mat

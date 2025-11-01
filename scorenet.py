import math
import torch
from torch import nn
from torch.nn import functional as F

import math
import torch
from torch import nn
from torch.nn import functional as F

class ScoreNet(nn.Module):
    """
    Drop-in pair scorer for Pix2Poly's OMN:
      • Input feats: [B, N(+1), C] (BOS/CLS at index 0)
      • Behavior: drop BOS, fold (y,x) pairs → [B, N, C]
      • Output: raw scores [B, N, N] (combine two heads upstream as P1 + P2^T)

    Adds:
      • Multi-head low-rank bilinear core (fast)
      • Tiny pair nonlinearity (separable MLP + outer product)
      • Soft geometry prior (distance RBF + cosθ/sinθ)  [if xy provided]
      • Angle harmonics (θ, 2θ, 4θ)                      [if xy provided]
      • PAD-aware masking for geometry biases
      • Bounded scale knobs for stability (sigmoid param)
    """
    def __init__(
        self,
        in_channels: int | None = None,   # None -> infer via LazyLinear (works for C=128/256)
        heads: int = 6,
        rank: int = 32,
        pair_hidden: int = 16,
        alpha_pair: float = 0.5,
        # Geometry / harmonics
        use_geometry: bool = True,
        use_angle_harmonics: bool = True,
        rbf_K: int = 8,
        alpha_geom: float = 0.6,
        alpha_angle: float = 0.6,
    ):
        super().__init__()
        self.H, self.R = heads, rank
        self.use_geometry = use_geometry
        self.use_angle_harmonics = use_angle_harmonics
        self.rbf_K = rbf_K

        # Projections (Lazy if C unknown)
        Linear = nn.LazyLinear if in_channels is None else nn.Linear
        self.q_proj = Linear(heads * rank, bias=False)     # C -> H*R
        self.k_proj = Linear(heads * rank, bias=False)     # C -> H*R
        self.head_scale = 1.0 / math.sqrt(rank)

        # Pair nonlinearity (separable, tiny)
        self.ph_i = Linear(pair_hidden, bias=True)         # C -> Dp
        self.ph_j = Linear(pair_hidden, bias=True)         # C -> Dp
        self._alpha_pair = nn.Parameter(torch.tensor(alpha_pair, dtype=torch.float32))

        # Soft geometry prior (RBF on distance + direction)
        if use_geometry:
            self.register_buffer("rbf_mu", torch.linspace(0.0, math.sqrt(2.0), steps=rbf_K))
            spacing = (self.rbf_mu[1] - self.rbf_mu[0]).item() if rbf_K > 1 else 1.0
            self.register_buffer("rbf_gamma", torch.tensor(1.0 / (2.0 * (spacing**2) + 1e-8)))
            # geom_feat = [dy, dx] + RBFs(r) + [cosθ, sinθ] -> scalar bias
            self.geom_mlp = nn.Sequential(
                nn.Linear(2 + rbf_K + 2, 32), nn.ReLU(inplace=True),
                nn.Linear(32, 1)
            )
            self._alpha_geom = nn.Parameter(torch.tensor(alpha_geom, dtype=torch.float32))

        # Angle harmonics (θ, 2θ, 4θ)
        if use_angle_harmonics:
            self.ang_mlp = nn.Sequential(
                nn.Linear(6, 16), nn.ReLU(inplace=True),
                nn.Linear(16, 1)
            )
            self._alpha_angle = nn.Parameter(torch.tensor(alpha_angle, dtype=torch.float32))

        # Diagonal bias (pads/self-loops) and bounded global scale
        self.diag_proj = Linear(1, bias=True)              # C -> 1
        self._logit_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    # ----- bounded scalars (keep logits well-conditioned) -----
    @staticmethod
    def _pos_scale(p: torch.Tensor, lo: float = 0.0, hi: float = 1.5) -> torch.Tensor:
        return lo + (hi - lo) * torch.sigmoid(p)

    @staticmethod
    def _logit_scale_fn(p: torch.Tensor, lo: float = 0.5, hi: float = 3.0) -> torch.Tensor:
        return lo + (hi - lo) * torch.sigmoid(p)

    # ----- folding: drop BOS/CLS and fold (y,x) pairs -> per-vertex features -----
    @staticmethod
    def _fold_vertices(feats: torch.Tensor) -> torch.Tensor:
        feats = feats[:, 1:]                                # [B, L-1, C]
        B, L, C = feats.shape
        return feats.view(B, L // 2, 2, C).mean(dim=2)      # [B, N, C]

    # ----- forward -----
    def forward(self, feats: torch.Tensor, xy: torch.Tensor | None = None) -> torch.Tensor:
        """
        feats: [B, N+1, C] (or [B, N, C] if upstream dropped BOS; we still drop index 0)
        xy   : [B, N, 2] in [0,1] (optional; enables geometry & harmonics)
        returns: [B, N, N] raw scores
        """
        x = self._fold_vertices(feats)                      # [B, N, C]
        B, N, C = x.shape

        # Ensure LazyLinear init (once)
        if isinstance(self.diag_proj, nn.LazyLinear):
            _ = self.diag_proj(x.mean(dim=1, keepdim=True))

        # (1) Multi-head low-rank bilinear core
        Q = self.q_proj(x).view(B, N, self.H, self.R)       # [B,N,H,R]
        K = self.k_proj(x).view(B, N, self.H, self.R)       # [B,N,H,R]
        S = torch.einsum('b n h r, b m h r -> b h n m', Q, K).sum(dim=1)  # [B,N,N]
        S = S / math.sqrt(self.R)

        # (2) Pair nonlinearity (separable MLP + outer product)
        a_i = F.relu(self.ph_i(x))                          # [B,N,Dp]
        b_j = F.relu(self.ph_j(x))                          # [B,N,Dp]
        S_pair = torch.einsum('b n d, b m d -> b n m', a_i, b_j)  # [B,N,N]
        S = S + self._pos_scale(self._alpha_pair) * S_pair

        # (3) Soft geometry prior + angle harmonics (if coords provided)
        if xy is not None:
            # coords in [0,1]
            yv, xv = xy[..., 0], xy[..., 1]                 # [B,N]
            dy = yv.unsqueeze(2) - yv.unsqueeze(1)          # [B,N,N]
            dx = xv.unsqueeze(2) - xv.unsqueeze(1)
            theta = torch.atan2(dy, dx)                     # [-π, π]

            # Heuristic PAD mask: PAD vertices commonly map to (0,0)
            pad_i = (yv.abs() < 1e-9) & (xv.abs() < 1e-9)   # [B,N]
            pad_pair = pad_i.unsqueeze(2) | pad_i.unsqueeze(1)  # [B,N,N]

            if self.use_geometry:
                r = torch.sqrt(dy * dy + dx * dx + 1e-8)    # [B,N,N] in [0, √2]
                phi = torch.exp(-self.rbf_gamma * (r.unsqueeze(-1) - self.rbf_mu)**2)  # [B,N,N,K]
                c1, s1 = torch.cos(theta), torch.sin(theta)
                geom_feat = torch.cat(
                    [dy.unsqueeze(-1), dx.unsqueeze(-1), phi, c1.unsqueeze(-1), s1.unsqueeze(-1)],
                    dim=-1
                )                                            # [B,N,N, 2+K+2]
                G = self.geom_mlp(geom_feat).squeeze(-1)     # [B,N,N]
                G = G.masked_fill(pad_pair, 0.0)
                S = S + self._pos_scale(self._alpha_geom) * G

            if self.use_angle_harmonics:
                c1, s1 = torch.cos(theta), torch.sin(theta)
                c2, s2 = torch.cos(2.0 * theta), torch.sin(2.0 * theta)
                c4, s4 = torch.cos(4.0 * theta), torch.sin(4.0 * theta)
                ang_feat = torch.stack([c1, s1, c2, s2, c4, s4], dim=-1)  # [B,N,N,6]
                Hb = self.ang_mlp(ang_feat).squeeze(-1)                   # [B,N,N]
                Hb = Hb.masked_fill(pad_pair, 0.0)
                S = S + self._pos_scale(self._alpha_angle) * Hb

        # (4) Diagonal bias (pads/self-loops) and bounded global scale
        S = S + torch.diag_embed(self.diag_proj(x).squeeze(-1))
        S = S * self._logit_scale_fn(self._logit_scale)

        return S



# In your EncoderDecoder.forward (teacher forcing), build xy:
seq = tgt[:, 1:]                  # drop BOS
N   = cfg.N_VERTICES
seq = seq[:, : 2*N]
y_ids = seq[:, 0::2].float(); x_ids = seq[:, 1::2].float()
denom = max(1, int(getattr(cfg, 'NUM_BINS', getattr(cfg, 'INPUT_HEIGHT', 224)) - 1))
xy = torch.stack([y_ids/denom, x_ids/denom], dim=-1)    # [B,N,2] in [0,1]

scores = self.scorenet1(feats, xy=xy)   # and similarly for scorenet2


# Stabilising larger ViT backbone:

# in Decoder.__init__(...)
self.encoder_pos_embed = nn.Parameter(torch.randn(1, encoder_len, dim) * .02)
self.encoder_pos_drop  = nn.Dropout(p=0.05)
self.memory_norm       = nn.LayerNorm(dim, eps=1e-6)  # <— add this line

# In decoder.forward, BEFORE (current code)
# encoder_out = self.encoder_pos_drop(
#     encoder_out + self.encoder_pos_embed
# )

# AFTER (apply norm, then add pos, then dropout)
encoder_out = self.memory_norm(encoder_out)                  # <— add this line
encoder_out = self.encoder_pos_drop(
    encoder_out + self.encoder_pos_embed
)


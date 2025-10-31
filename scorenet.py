import math
import torch
from torch import nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------
# ScoreNet (Drop-in, Efficient + Pair Nonlinearity + Soft Geometry + Harmonics)
# -----------------------------------------------------------------------------
# Preserves original behavior:
#   • Input feats: [B, N(+1), C] (first token is BOS/CLS)
#   • Folding: drop BOS, fold (y,x) token pairs → [B, N, C]
#   • Output: raw scores [B, N, N] (no antisymmetry, no diag mask)
#
# Additions:
#   1) Pair nonlinearity: separable MLP on vertices (f_i -> a_i, f_j -> b_j),
#      then outer-product S_pair(i,j) = <a_i, b_j> (recovers conv-head gating).
#   2) Soft geometry prior (requires xy): RBF(r) on distance + [cosθ, sinθ]
#   3) Angle harmonics (requires xy): [cosθ, sinθ, cos2θ, sin2θ, cos4θ, sin4θ]
#      Optionally made relative to a per-image dominant orientation θ0 (PCA).
#
# Notes:
#   • xy is optional for strict drop-in; pass xy to turn geometry/harmonics on.
#   • Uses LazyLinear when in_channels=None, so it adapts to C=128/256 automatically.
# -----------------------------------------------------------------------------

class ScoreNet(nn.Module):
    def __init__(
        self,
        in_channels: int | None = None,   # None -> infer via LazyLinear
        heads: int = 6,
        rank: int = 32,
        pair_hidden: int = 16,
        alpha_pair: float = 0.5,
        # Geometry / harmonics
        use_geometry: bool = True,
        use_angle_harmonics: bool = True,
        use_dominant_orientation: bool = True,  # set False to use absolute θ
        rbf_K: int = 8,
        alpha_geom: float = 0.6,
        alpha_angle: float = 0.6,
    ):
        super().__init__()
        self.H, self.R = heads, rank
        self.use_geometry = use_geometry
        self.use_angle_harmonics = use_angle_harmonics
        self.use_dominant_orientation = use_dominant_orientation
        self.rbf_K = rbf_K

        # --- Projections (Lazy if C unknown) ---
        Linear = nn.LazyLinear if in_channels is None else nn.Linear
        self.q_proj = Linear(heads * rank, bias=False)     # C -> H*R
        self.k_proj = Linear(heads * rank, bias=False)     # C -> H*R
        self.head_scale = 1.0 / math.sqrt(rank)

        # --- Pair nonlinearity (separable, tiny) ---
        self.ph_i = Linear(pair_hidden, bias=True)         # C -> Dp
        self.ph_j = Linear(pair_hidden, bias=True)         # C -> Dp
        self.alpha_pair = nn.Parameter(torch.tensor(alpha_pair, dtype=torch.float32))

        # --- Soft geometry prior (RBF on r + direction) ---
        if use_geometry:
            # Distance RBF centers over [0, √2] (coords expected in [0,1])
            self.register_buffer("rbf_mu", torch.linspace(0.0, math.sqrt(2.0), steps=rbf_K))
            spacing = (self.rbf_mu[1] - self.rbf_mu[0]).item() if rbf_K > 1 else 1.0
            self.register_buffer("rbf_gamma", torch.tensor(1.0 / (2.0 * (spacing**2) + 1e-8)))
            # geom_feat = [dy, dx] + RBFs(r) + [cosθ, sinθ]  → scalar
            self.geom_mlp = nn.Sequential(
                nn.Linear(2 + rbf_K + 2, 32), nn.ReLU(inplace=True),
                nn.Linear(32, 1)
            )
            self.alpha_geom = nn.Parameter(torch.tensor(alpha_geom, dtype=torch.float32))

        # --- Angle harmonics (θ, 2θ, 4θ) ---
        if use_angle_harmonics:
            self.ang_mlp = nn.Sequential(
                nn.Linear(6, 16), nn.ReLU(inplace=True),
                nn.Linear(16, 1)
            )
            self.alpha_angle = nn.Parameter(torch.tensor(alpha_angle, dtype=torch.float32))

        # Diagonal bias (pads/self-loops)
        self.diag_proj = Linear(1, bias=True)              # C -> 1

        # Global temperature for logits
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _fold_vertices(feats: torch.Tensor) -> torch.Tensor:
        # Drop BOS, fold (y,x) -> 1 vertex by mean
        feats = feats[:, 1:]
        B, L, C = feats.shape
        return feats.view(B, L // 2, 2, C).mean(dim=2)     # [B, N, C]

    @staticmethod
    def _principal_orientation(xy: torch.Tensor) -> torch.Tensor:
        """
        Estimate per-image dominant orientation θ0 via PCA on [B,N,2] coords.
        Returns θ0: [B,1,1] radians.
        """
        B, N, _ = xy.shape
        X = xy - xy.mean(dim=1, keepdim=True)              # [B,N,2]
        C = torch.einsum('b n d, b n e -> b d e', X, X) / (N + 1e-8)  # [B,2,2]
        v = torch.randn(B, 2, 1, device=xy.device, dtype=xy.dtype)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
        for _ in range(5):
            v = torch.matmul(C, v)
            v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
        theta0 = torch.atan2(v[:, 1, 0], v[:, 0, 0])       # [B]
        return theta0.view(B, 1, 1)

    def forward(self, feats: torch.Tensor, xy: torch.Tensor | None = None) -> torch.Tensor:
        """
        feats: [B, N+1, C] or [B, N, C]
        xy   : [B, N, 2] in [0,1] (optional; enables geometry/harmonics)
        returns: [B, N, N] scores
        """
        x = self._fold_vertices(feats)                     # [B, N, C]
        B, N, C = x.shape

        # Ensure LazyLinear shapes are set (one cheap pass)
        if isinstance(self.diag_proj, nn.LazyLinear):
            _ = self.diag_proj(x.mean(dim=1, keepdim=True))  # [B,1,1]

        # --- (1) Multi-head low-rank bilinear core ---
        Q = self.q_proj(x).view(B, N, self.H, self.R)      # [B,N,H,R]
        K = self.k_proj(x).view(B, N, self.H, self.R)      # [B,N,H,R]
        S = torch.einsum('b n h r, b m h r -> b h n m', Q, K) * self.head_scale
        S = S.sum(dim=1)                                   # [B, N, N]

        # --- (2) Pair nonlinearity (separable MLP + outer product) ---
        a_i = F.relu(self.ph_i(x))                         # [B, N, Dp]
        b_j = F.relu(self.ph_j(x))                         # [B, N, Dp]
        S_pair = torch.einsum('b n d, b m d -> b n m', a_i, b_j)  # [B, N, N]
        S = S + self.alpha_pair.clamp_min(0.0) * S_pair

        # --- (3) Soft geometry prior + angle harmonics (if coords provided) ---
        if xy is not None:
            yv, xv = xy[..., 0], xy[..., 1]                # [B,N]
            dy = yv.unsqueeze(2) - yv.unsqueeze(1)         # [B,N,N]
            dx = xv.unsqueeze(2) - xv.unsqueeze(1)
            theta = torch.atan2(dy, dx)                    # [-π, π]
            if self.use_dominant_orientation:
                theta0 = self._principal_orientation(xy)   # [B,1,1]
                # wrap angle relative to θ0 back into [-π, π]
                theta = torch.atan2(torch.sin(theta - theta0),
                                     torch.cos(theta - theta0))

            # (3a) Soft geometry prior: distance RBF + direction
            if self.use_geometry:
                r = torch.sqrt(dy * dy + dx * dx + 1e-8)   # [B,N,N] in [0, √2]
                phi = torch.exp(-self.rbf_gamma * (r.unsqueeze(-1) - self.rbf_mu)**2)  # [B,N,N,K]
                c1, s1 = torch.cos(theta), torch.sin(theta)
                geom_feat = torch.cat([dy.unsqueeze(-1), dx.unsqueeze(-1), phi,
                                       c1.unsqueeze(-1), s1.unsqueeze(-1)], dim=-1)    # [B,N,N,2+K+2]
                G = self.geom_mlp(geom_feat).squeeze(-1)   # [B,N,N]
                S = S + self.alpha_geom.clamp_min(0.0) * G

            # (3b) Angle harmonics (right-angle-friendly)
            if self.use_angle_harmonics:
                c1, s1 = torch.cos(theta), torch.sin(theta)
                c2, s2 = torch.cos(2.0 * theta), torch.sin(2.0 * theta)
                c4, s4 = torch.cos(4.0 * theta), torch.sin(4.0 * theta)
                ang_feat = torch.stack([c1, s1, c2, s2, c4, s4], dim=-1)  # [B,N,N,6]
                Hbias = self.ang_mlp(ang_feat).squeeze(-1)                # [B,N,N]
                S = S + self.alpha_angle.clamp_min(0.0) * Hbias

        # --- (4) Diagonal bias (pads) ---
        S = S + torch.diag_embed(self.diag_proj(x).squeeze(-1))

        return S * self.logit_scale.clamp_min(1e-2)


# In your EncoderDecoder.forward (teacher forcing), build xy:
seq = tgt[:, 1:]                  # drop BOS
N   = cfg.N_VERTICES
seq = seq[:, : 2*N]
y_ids = seq[:, 0::2].float(); x_ids = seq[:, 1::2].float()
denom = max(1, int(getattr(cfg, 'NUM_BINS', getattr(cfg, 'INPUT_HEIGHT', 224)) - 1))
xy = torch.stack([y_ids/denom, x_ids/denom], dim=-1)    # [B,N,2] in [0,1]

scores = self.scorenet1(feats, xy=xy)   # and similarly for scorenet2


# Extra norm after encoder forward:
enc_tokens = encoder(images)                    # [B, S, Cenc]

# 1a) Project to decoder dim if Cenc != Cdec (you likely already do this)
enc_tokens = self.enc_proj(enc_tokens)          # [B, S, Cdec]

# 1b) NEW: normalize to tame scale drift
self.enc_norm = getattr(self, "enc_norm", nn.LayerNorm(enc_tokens.size(-1)).to(enc_tokens.device))
enc_tokens = self.enc_norm(enc_tokens)          # [B, S, Cdec]

# pass enc_tokens to decoder cross-attention
decoder_outputs = decoder(tgt_tokens, memory=enc_tokens)

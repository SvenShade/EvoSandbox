import math
import torch
from torch import nn
from torch.nn import functional as F


class ScoreNet(nn.Module):
    """
    Drop-in pair scorer for Pix2Poly's OMN:
      • Input feats: [B, N(+1), C] (BOS/CLS at index 0), where C == in_channels.
      • Behavior: drop BOS, fold (y,x) pairs → [B, N, C]
      • Output: raw scores [B, N, N] (combine two heads upstream as P1 + P2^T)

    Adds:
      • Multi-head low-rank bilinear core (fast)
      • Tiny pair nonlinearity (separable MLP + outer product)
      • Soft geometry prior (distance RBF + cosθ/sinθ)  [if xy provided]
      • Angle harmonics (θ, 2θ, 4θ)                      [if xy provided]
      • PAD/EOS-aware masking for geometry via `mask` ([B,N] bool)
      • Bounded scale knobs for stability (sigmoid param)
    """
    def __init__(
        self,
        in_channels: int,                # REQUIRED: decoder hidden size C
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

        # Projections
        self.q_proj = nn.Linear(in_channels, heads * rank, bias=False)  # C -> H*R
        self.k_proj = nn.Linear(in_channels, heads * rank, bias=False)  # C -> H*R
        self.head_scale = 1.0 / math.sqrt(rank)

        # Pair nonlinearity (separable, tiny)
        self.ph_i = nn.Linear(in_channels, pair_hidden, bias=True)      # C -> Dp
        self.ph_j = nn.Linear(in_channels, pair_hidden, bias=True)      # C -> Dp
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
        self.diag_proj = nn.Linear(in_channels, 1, bias=True)           # C -> 1
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
    def forward(
        self,
        feats: torch.Tensor,
        xy: torch.Tensor | None = None,       # [B, N, 2] in [0,1]
        mask: torch.Tensor | None = None,     # [B, N] bool; True where vertex is valid
    ) -> torch.Tensor:
        """
        feats: [B, N+1, C] (or [B, N, C] if upstream dropped BOS; we still drop index 0)
        xy   : [B, N, 2] in [0,1] (optional; enables geometry & harmonics)
        mask : [B, N] bool (optional; zeros geometry bias for invalid PAD/EOS vertices)
        returns: [B, N, N] raw scores
        """
        x = self._fold_vertices(feats)                      # [B, N, C]
        B, N, C = x.shape

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
            yv, xv = xy[..., 0], xy[..., 1]                 # [B,N]
            dy = yv.unsqueeze(2) - yv.unsqueeze(1)          # [B,N,N]
            dx = xv.unsqueeze(2) - xv.unsqueeze(1)
            theta = torch.atan2(dy, dx)                     # [-π, π]

            # Invalid-pair mask grid from PAD/EOS vertex mask
            pair_invalid = None
            if mask is not None:
                invalid = (~mask.bool())
                pair_invalid = invalid.unsqueeze(2) | invalid.unsqueeze(1)  # [B,N,N]

            if self.use_geometry:
                r = torch.sqrt(dy * dy + dx * dx + 1e-8)    # [B,N,N] in [0, √2]
                phi = torch.exp(-self.rbf_gamma * (r.unsqueeze(-1) - self.rbf_mu)**2)  # [B,N,N,K]
                c1, s1 = torch.cos(theta), torch.sin(theta)
                geom_feat = torch.cat(
                    [dy.unsqueeze(-1), dx.unsqueeze(-1), phi, c1.unsqueeze(-1), s1.unsqueeze(-1)],
                    dim=-1
                )                                            # [B,N,N, 2+K+2]
                G = self.geom_mlp(geom_feat).squeeze(-1)     # [B,N,N]
                if pair_invalid is not None:
                    G = G.masked_fill(pair_invalid, 0.0)
                S = S + self._pos_scale(self._alpha_geom) * G

            if self.use_angle_harmonics:
                c1, s1 = torch.cos(theta), torch.sin(theta)
                c2, s2 = torch.cos(2.0 * theta), torch.sin(2.0 * theta)
                c4, s4 = torch.cos(4.0 * theta), torch.sin(4.0 * theta)
                ang_feat = torch.stack([c1, s1, c2, s2, c4, s4], dim=-1)  # [B,N,N,6]
                Hb = self.ang_mlp(ang_feat).squeeze(-1)                   # [B,N,N]
                if pair_invalid is not None:
                    Hb = Hb.masked_fill(pair_invalid, 0.0)
                S = S + self._pos_scale(self._alpha_angle) * Hb

        # (4) Diagonal bias (pads/self-loops) and bounded global scale
        S = S + torch.diag_embed(self.diag_proj(x).squeeze(-1))
        S = S * self._logit_scale_fn(self._logit_scale)

        return S




# utils.py
import torch

def build_xy_and_mask_from_tokens(tgt: torch.Tensor,
                                  pad_id: int,
                                  num_bins_h: int,
                                  num_bins_w: int,
                                  n_vertices: int):
    """
    tgt: [B, L] Long (BOS at index 0; then (y,x) repeated; then EOS; then PAD)
    Returns:
      xy   : [B, N, 2] in [0,1] (y, x)
      mask : [B, N] boolean, True where vertex is valid
    Notes:
      • Uses *the same* dequantization convention as your Tokenizer.dequantize:
        id / (num_bins-1). No bin-centering offset is added (keeps it consistent).
      • EOS handling: any pair that straddles or follows EOS is masked out.
      • PAD handling: any pair containing PAD is masked out.
    """
    B, L = tgt.shape
    device = tgt.device
    N = n_vertices

    # Drop BOS; take at most 2*N tokens (the model folds pairs in ScoreNet)
    seq = tgt[:, 1:]                            # [B, L-1]
    if seq.size(1) < 2 * N:
        # pad with PAD to avoid view errors in upstream folding (mirrors collate padding behavior)
        pad = torch.full((B, 2 * N - seq.size(1)), pad_id, device=device, dtype=seq.dtype)
        seq = torch.cat([seq, pad], dim=1)
    else:
        seq = seq[:, : 2 * N]                   # [B, 2N]

    # Reshape into pairs (y, x)
    pairs = seq.view(B, N, 2)                   # [B, N, 2]
    y_ids = pairs[..., 0]
    x_ids = pairs[..., 1]

    # EOS/PAD-aware mask:
    # A pair is valid iff both tokens are < PAD and != PAD, and appear before the first EOS.
    # 1) PAD mask
    non_pad = (y_ids != pad_id) & (x_ids != pad_id)        # [B, N]

    # 2) EOS position (first EOS in the *flat* sequence after BOS)
    #    Pairs at/after EOS are invalid.
    #    We rebuild EOS over seq to remain consistent with your postprocess() logic.
    eos_code = pad_id - 1  # since PAD = num_bins+2, EOS = num_bins+1
    # mark EOS positions in [B, 2N]
    is_eos = (seq == eos_code)                              # [B, 2N]
    # first EOS index per batch (if none, set to 2N)
    first_eos = torch.where(is_eos.any(dim=1),
                            is_eos.float().argmax(dim=1),
                            torch.full((B,), 2 * N, device=device, dtype=torch.long))   # [B]
    # pair index (0..N-1) corresponds to positions (2*i, 2*i+1) in seq
    pair_idx = torch.arange(N, device=device).view(1, N) * 2  # [1, N] -> tokens positions
    # valid if both token positions are strictly before first EOS
    before_eos = (pair_idx + 1) < first_eos.view(B, 1)        # [B, N]

    mask = non_pad & before_eos                                # [B, N]

    # Dequantize exactly like Tokenizer.dequantize
    den_y = max(1, int(num_bins_h - 1))
    den_x = max(1, int(num_bins_w - 1))
    y = (y_ids.clamp_min(0).float() / den_y).clamp(0.0, 1.0)
    x = (x_ids.clamp_min(0).float() / den_x).clamp(0.0, 1.0)

    # Zero-out invalid pairs (optional; the mask is what downstream should rely on)
    y = torch.where(mask, y, torch.zeros_like(y))
    x = torch.where(mask, x, torch.zeros_like(x))

    xy = torch.stack([y, x], dim=-1)                          # [B, N, 2]
    return xy, mask


# engine.py (train_one_epoch / valid_one_epoch), after:
y_input = y[:, :-1]
y_expected = y[:, 1:]

# Build geometry from teacher-forced tokens
xy, pair_mask = build_xy_and_mask_from_tokens(
    y_input,
    pad_id=CFG.PAD_IDX,
    num_bins_h=CFG.NUM_BINS,        # height bins
    num_bins_w=CFG.NUM_BINS,        # width bins (your CFG ties these)
    n_vertices=CFG.N_VERTICES
)

# If your EncoderDecoder.forward accepts xy/mask and passes them to scorenets:
preds, perm_mat = model(x, y_input, xy=xy, mask=pair_mask)


# utils.py::test_generate (right before calling scorenets)
# Build geometry from generated tokens
with torch.no_grad():
    xy, pair_mask = build_xy_and_mask_from_tokens(
        batch_preds,                     # [B, 1+2N] incl BOS
        pad_id=tokenizer.PAD_code,
        num_bins_h=tokenizer.num_bins,
        num_bins_w=tokenizer.num_bins,
        n_vertices=CFG.N_VERTICES
    )

if isinstance(model, torch.nn.parallel.DistributedDataParallel):
    m = model.module
else:
    m = model

# If using geometry-aware scorenets:
perm_scores = m.scorenet1(feats, xy=xy, mask=pair_mask) + m.scorenet2(feats, xy=xy, mask=pair_mask).transpose(1, 2)

# Else (original heads):
# perm_scores = m.scorenet1(feats) + m.scorenet2(feats).transpose(1, 2)

perm_preds = scores_to_permutations(perm_scores)

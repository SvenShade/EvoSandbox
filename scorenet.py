import torch
from torch import nn
from torch.nn import functional as F
import math

# --- New head: Antisymmetric matcher (drop-in replacement for two ScoreNet heads) ---
class AntisymMatcher(nn.Module):
    """
    Multi-head antisymmetric matcher for building footprints.

    Computes pairwise successor logits S ∈ R^{B×N×N} from decoder token features
    using H low-rank bilinear heads, then enforces antisymmetry by construction:
        S = 0.5 * (A - A^T)
    where per-head scores are A_h(i,j) = (Q_h f_i)^T (K_h f_j). Heads are summed.

    Designed for buildings (single-ring), so we suppress the diagonal (no self-edges).
    Padding is still handled by the external OT "dustbin" via bin_score.
    """
    def __init__(self, n_vertices: int, in_channels: int = 256, rank: int = 16, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_vertices = n_vertices
        self.in_channels = in_channels
        self.rank = rank              # per-head rank
        self.heads = heads            # number of low-rank heads

        # Low-rank projections for all heads in one matmul (C -> H*R)
        self.q_proj = nn.Linear(in_channels, rank * heads, bias=False)
        self.k_proj = nn.Linear(in_channels, rank * heads, bias=False)

        # Stabilization
        self.do = nn.Dropout(dropout)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.head_scale = 1.0 / math.sqrt(rank)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        feats = feats[:, 1:]
        B, L, C = feats.shape
        # Fold every two tokens into one vertex feature (same assumption as ScoreNet)
        feats = feats.view(B, L // 2, 2, C).mean(dim=2) # [B, N, C], N = L//2
        N = feats.size(1)
        # Lightweight normalization that adapts to C
        x = F.layer_norm(feats, (C,)))
        x = self.do(x)
            
        # If decoder produced more vertices than target_N, keep first target_N
        if N > target_N:
            x = x[:, :target_N, :]
            N = target_N
        
        
        # Projections for multi-head low-rank bilinear scoring
        Q = self.q_proj(x).view(B, N, self.heads, self.rank)
        K = self.k_proj(x).view(B, N, self.heads, self.rank)
        
        
        # Per-head bilinear scores, then sum heads → [B, N, N]
        # A_h(i,j) = <Q[i,h,:], K[j,h,:]>
        A = torch.einsum('b n h r, b m h r -> b h n m', Q, K)
        A = A * self.head_scale
        A = A.sum(dim=1) # sum over heads → [B, N, N]
        
        
        # Enforce antisymmetry; suppress diagonal (no self loops)
        S = 0.5 * (A - A.transpose(1, 2))
        # If fewer than max vertices, pad logits to [N_cfg,N_cfg] with very negative values
        if N < self.n_vertices:
        pad = self.n_vertices - N
        S = F.pad(S, (0, pad, 0, pad), value=-1e4)
        N = self.n_vertices
        diag_mask = torch.eye(N, device=S.device, dtype=torch.bool).unsqueeze(0)
        S = S.masked_fill(diag_mask, float('-inf'))
        
        
        # Optional temperature
        S = S * self.logit_scale.clamp_min(1e-2)
        return S


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

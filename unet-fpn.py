import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class UNetFPNBackbone(nn.Module):
    """
    Efficient drop-in backbone to replace vit_small_patch8_224.

    - Input:  x [B, 3, H, W], H,W divisible by 8 (e.g. 224x224)
    - Output: tokens [B, (H/8)*(W/8), 256]
      For 224x224 -> [B, 784, 256]

    Design:
      - 3-stage lightweight encoder with strides: 1, 2, 4, 8
      - FPN-style fusion of multi-scale features onto stride-8 grid
      - Modest channel sizes to keep params/FLOPs low
    """
    def __init__(self,
                 in_chans: int = 3,
                 embed_dim: int = 256,
                 base_ch: int = 32):
        super().__init__()

        # Encoder channel config (small & efficient)
        c1 = base_ch          # 32
        c2 = base_ch * 2      # 64
        c3 = base_ch * 4      # 128

        # Stage 1: stride 1 (H, W)
        self.enc1 = nn.Sequential(
            ConvBNAct(in_chans, c1, k=3, s=1, p=1),
            ConvBNAct(c1, c1, k=3, s=1, p=1),
        )

        # Down 1: stride 2 (H/2, W/2)
        self.down1 = ConvBNAct(c1, c2, k=3, s=2, p=1)

        # Stage 2: (H/2, W/2)
        self.enc2 = nn.Sequential(
            ConvBNAct(c2, c2, k=3, s=1, p=1),
            ConvBNAct(c2, c2, k=3, s=1, p=1),
        )

        # Down 2: stride 4 (H/4, W/4)
        self.down2 = ConvBNAct(c2, c3, k=3, s=2, p=1)

        # Stage 3: (H/4, W/4)
        self.enc3 = nn.Sequential(
            ConvBNAct(c3, c3, k=3, s=1, p=1),
            ConvBNAct(c3, c3, k=3, s=1, p=1),
        )

        # Down 3: stride 8 (H/8, W/8)
        self.down3 = ConvBNAct(c3, c3, k=3, s=2, p=1)

        # FPN lateral projections (all → embed_dim)
        self.lat1 = nn.Conv2d(c1, embed_dim, kernel_size=1, bias=False)  # from H
        self.lat2 = nn.Conv2d(c2, embed_dim, kernel_size=1, bias=False)  # from H/2
        self.lat3 = nn.Conv2d(c3, embed_dim, kernel_size=1, bias=False)  # from H/4
        self.lat4 = nn.Conv2d(c3, embed_dim, kernel_size=1, bias=False)  # from H/8

        # Lightweight refinement on fused stride-8 map
        self.refine = nn.Sequential(
            ConvBNAct(embed_dim, embed_dim, k=3, s=1, p=1),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W], H,W divisible by 8

        Returns:
            tokens: [B, (H/8)*(W/8), 256]
        """
        B, _, H, W = x.shape

        # Encoder path
        x1 = self.enc1(x)              # [B, c1, H,   W]
        x2 = self.enc2(self.down1(x1)) # [B, c2, H/2, W/2]
        x3 = self.enc3(self.down2(x2)) # [B, c3, H/4, W/4]
        x4 = self.down3(x3)            # [B, c3, H/8, W/8]

        H8, W8 = x4.shape[2], x4.shape[3]

        # FPN-style fusion onto stride-8 grid (cheap: mostly 1x1 + pooling)
        p4 = self.lat4(x4)                                # [B, D, H/8,   W/8]
        p3 = F.adaptive_avg_pool2d(self.lat3(x3), (H8, W8))
        p2 = F.adaptive_avg_pool2d(self.lat2(x2), (H8, W8))
        p1 = F.adaptive_avg_pool2d(self.lat1(x1), (H8, W8))

        fused = p4 + p3 + p2 + p1                         # [B, D, H/8, W/8]
        fused = self.refine(fused)                        # [B, D, H/8, W/8]

        # Flatten to token sequence
        tokens = fused.flatten(2).transpose(1, 2)         # [B, H8*W8, D]

        return tokens

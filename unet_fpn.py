import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNetFPN(nn.Module):
    """
    Input:  (B, 3, 224, 224)
    Output: (B, 196, 192)
    """
    def __init__(self, embed_dim=192, widths=(64,128,256,256), use_ln=True):
        super().__init__()
        c1,c2,c3,c4 = widths
        # encoder: 224->112->56->28->14
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(c1, c2))  # 112->56
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(c2, c3))  # 56->28
        self.enc4 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(c3, c4))  # 28->14

        # laterals to D
        self.lat1 = nn.Conv2d(c1, embed_dim, 1, bias=False)  # 112x112
        self.lat2 = nn.Conv2d(c2, embed_dim, 1, bias=False)  # 56x56
        self.lat3 = nn.Conv2d(c3, embed_dim, 1, bias=False)  # 28x28
        self.lat4 = nn.Conv2d(c4, embed_dim, 1, bias=False)  # 14x14

        self.smooth = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True),
        )
        self.ln = nn.LayerNorm(embed_dim) if use_ln else nn.Identity()

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0); nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        assert x.shape[1:] == (3,224,224), "Expected (B,3,224,224)."
        f1 = self.stem(x)     # (B,c1,112,112)
        f2 = self.enc2(f1)    # (B,c2,56,56)
        f3 = self.enc3(f2)    # (B,c3,28,28)
        f4 = self.enc4(f3)    # (B,c4,14,14)

        p4 = self.lat4(f4)
        p3 = F.interpolate(self.lat3(f3), size=p4.shape[-2:], mode="bilinear", align_corners=False)
        p2 = F.interpolate(self.lat2(f2), size=p4.shape[-2:], mode="bilinear", align_corners=False)
        p1 = F.interpolate(self.lat1(f1), size=p4.shape[-2:], mode="bilinear", align_corners=False)

        fused = self.smooth(p1 + p2 + p3 + p4)       # (B, D, 14, 14)
        tokens = fused.flatten(2).transpose(1, 2)    # (B, 196, 192)
        return self.ln(tokens)

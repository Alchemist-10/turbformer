import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Resize g1 to match x1 if needed
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        psi = F.relu(g1 + x1)
        psi = torch.sigmoid(self.psi(psi))
        return x * psi


class ProgressiveDilatedBlock(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, in_c, 3, padding=2, dilation=2)
        self.conv2 = nn.Conv2d(in_c, in_c, 3, padding=4, dilation=4)
        self.conv3 = nn.Conv2d(in_c, in_c, 3, padding=6, dilation=6)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        return x


class OAMRestoreNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=2, ablation_no_att=False, ablation_no_dil=False):
        super().__init__()
        self.ablation_no_att = ablation_no_att
        self.ablation_no_dil = ablation_no_dil

        # Timm returns 4 scale levels for swin_tiny_patch4_window7_224
        self.encoder = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, features_only=True)
        self.stem = nn.Conv2d(in_ch, 3, kernel_size=3, padding=1)

        # Decoder components
        # Swin feature dimensions: 96, 192, 384, 768
        self.up1 = nn.PixelShuffle(2)  # 768 -> 192
        self.conv1 = nn.Conv2d(192 + 384, 384, 3, padding=1)
        self.att1 = AttentionGate(192, 384, 192) if not ablation_no_att else None

        self.up2 = nn.PixelShuffle(2)  # 384 -> 96
        self.conv2 = nn.Conv2d(96 + 192, 192, 3, padding=1)
        self.att2 = AttentionGate(96, 192, 96) if not ablation_no_att else None

        self.up3 = nn.PixelShuffle(2)  # 192 -> 48
        self.conv3 = nn.Conv2d(48 + 96, 96, 3, padding=1)
        self.att3 = AttentionGate(48, 96, 48) if not ablation_no_att else None

        # Final upsample to original resolution (stride 4 patch -> 2x PixelShuffle -> 2x Upsample)
        self.final_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.refine = ProgressiveDilatedBlock(96) if not ablation_no_dil else nn.Conv2d(96, 96, 3, padding=1)
        self.head = nn.Conv2d(96, out_ch, 3, padding=1)

    def forward(self, x):
        x = self.stem(x)
        features = self.encoder(x)
        f1, f2, f3, f4 = features  # H/4 (96), H/8 (192), H/16 (384), H/32 (768)

        # Permute from timm's output (B, H, W, C) to (B, C, H, W)
        f1 = f1.permute(0, 3, 1, 2)
        f2 = f2.permute(0, 3, 1, 2)
        f3 = f3.permute(0, 3, 1, 2)
        f4 = f4.permute(0, 3, 1, 2)

        d1 = self.up1(f4)
        if not self.ablation_no_att:
            f3 = self.att1(g=d1, x=f3)
        d1 = F.relu(self.conv1(torch.cat([d1, f3], dim=1)))

        d2 = self.up2(d1)
        if not self.ablation_no_att:
            f2 = self.att2(g=d2, x=f2)
        d2 = F.relu(self.conv2(torch.cat([d2, f2], dim=1)))

        d3 = self.up3(d2)
        if not self.ablation_no_att:
            f1 = self.att3(g=d3, x=f1)
        d3 = F.relu(self.conv3(torch.cat([d3, f1], dim=1)))

        out = self.final_up(d3)
        out = self.refine(out)
        out = self.head(out)
        return out
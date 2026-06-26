import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    def __init__(self, skip_ch: int, gate_ch: int, inter_ch: int):
        super().__init__()
        self.W_g = nn.Conv2d(gate_ch, inter_ch, kernel_size=1, bias=False)
        self.W_x = nn.Conv2d(skip_ch, inter_ch, kernel_size=1, bias=False)
        self.psi = nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True)

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        g = self.W_g(gate)
        x = self.W_x(skip)

        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)

        attn = torch.sigmoid(self.psi(F.relu(g + x, inplace=True)))
        return skip * attn


class ProgressiveDilatedBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=4, dilation=4, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)

        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=6, dilation=6, bias=False)
        self.norm3 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        x = self.act(self.norm3(self.conv3(x)))
        return x + residual


class DecoderStage(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention
        self.reduce = ConvNormAct(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

        if use_attention:
            self.att = AttentionGate(
                skip_ch=skip_ch,
                gate_ch=out_ch,
                inter_ch=max(8, min(skip_ch, out_ch) // 2),
            )
        else:
            self.att = None

        self.fuse = nn.Sequential(
            ConvNormAct(out_ch + skip_ch, out_ch),
            ConvNormAct(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)

        if self.att is not None:
            skip = self.att(skip, x)

        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class OAMRestoreNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 2,
        pretrained: bool = True,
        ablation_no_att: bool = False,
        ablation_no_dil: bool = False,
    ):
        super().__init__()

        self.encoder = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=pretrained,
        features_only=True,
        in_chans=in_ch,
        out_indices=(0, 1, 2, 3),
)

        enc_chs = self.encoder.feature_info.channels()  # usually [96, 192, 384, 768]

        self.stem = ConvNormAct(in_ch, in_ch, kernel_size=3, stride=1, padding=1)

        self.bottleneck = nn.Sequential(
            ConvNormAct(enc_chs[3], 384, kernel_size=1, stride=1, padding=0),
            ConvNormAct(384, 384, kernel_size=3, stride=1, padding=1),
        )

        self.dec3 = DecoderStage(384, enc_chs[2], 256, use_attention=not ablation_no_att)
        self.dec2 = DecoderStage(256, enc_chs[1], 128, use_attention=not ablation_no_att)
        self.dec1 = DecoderStage(128, enc_chs[0], 96, use_attention=not ablation_no_att)

        if ablation_no_dil:
            self.refine = nn.Sequential(
                ConvNormAct(96, 96),
                ConvNormAct(96, 96),
            )
        else:
            self.refine = ProgressiveDilatedBlock(96)

        self.head = nn.Conv2d(96, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.shape[-2:]

        # 1. Process through the stem layout
        x = self.stem(x)

        # 2. Extract features from Swin backbone
        raw_features = self.encoder(x)

        # 3. 🚀 FIX: Convert Swin's NHWC layout to standard CNN NCHW format
        formatted_features = []
        for i, feat in enumerate(raw_features):
            # Check if channels are at the trailing axis
            if len(feat.shape) == 4 and feat.shape[1] != self.encoder.feature_info.channels()[i]:
                feat = feat.permute(0, 3, 1, 2).contiguous()
            formatted_features.append(feat)

        f1, f2, f3, f4 = formatted_features

        # 4. Decode and reconstruct phase maps smoothly
        x = self.bottleneck(f4)
        x = self.dec3(x, f3)
        x = self.dec2(x, f2)
        x = self.dec1(x, f1)

        x = F.interpolate(x, size=orig_size, mode="bilinear", align_corners=False)
        x = self.refine(x)
        out = self.head(x)

        return out
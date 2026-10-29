import torch
import torch.nn as nn
import torch.nn.functional as F
from model import LightweightUNet, DecoderBlock


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling for a small (8x8) bottleneck.
    Dilations 1, 2, 4 (6/12 would mostly sample padding at 8x8) + global pooling."""
    def __init__(self, in_ch=1280, out_ch=256, branch_ch=128, dilations=(1, 2, 4)):
        super().__init__()
        self.branches = nn.ModuleList()
        for d in dilations:
            k = 1 if d == 1 else 3
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_ch, branch_ch, k, padding=0 if d == 1 else d, dilation=d, bias=False),
                nn.BatchNorm2d(branch_ch),
                nn.ReLU(inplace=True),
            ))
        # global-context branch (no BatchNorm, so it is safe with batch size 1)
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, branch_ch, 1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(branch_ch * (len(dilations) + 1), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        g = self.gap(x)
        feats.append(F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False))
        return self.project(torch.cat(feats, dim=1))


class ASPPUNet(LightweightUNet):
    """LightweightUNet + ASPP on the bottleneck. Everything else is identical."""
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__(num_classes=num_classes, pretrained=pretrained)
        self.aspp = ASPP(in_ch=1280, out_ch=256)
        # dec4 now receives the 256-channel ASPP output instead of the raw 1280 channels
        self.dec4 = DecoderBlock(256, 96, 256)

    def forward(self, x):
        s1 = self.enc_stage1(x)
        s2 = self.enc_stage2(s1)
        s3 = self.enc_stage3(s2)
        s4 = self.enc_stage4(s3)
        bottleneck = self.aspp(self.enc_stage5(s4))

        d4 = self.dec4(bottleneck, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        out = self.final_upsample(d1)
        out = self.final_conv(out)
        return torch.sigmoid(out)


if __name__ == "__main__":
    m = ASPPUNet(pretrained=False)
    print("Parameters:", sum(p.numel() for p in m.parameters()))
    print("Output shape:", m(torch.randn(2, 1, 256, 256)).shape)
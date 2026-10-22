import torch
import torch.nn as nn
import torchvision.models as models


class SEBlock(nn.Module):
    """Squeeze-and-Excitation attention block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DecoderBlock(nn.Module):
    """Upsample + conv block for the decoder path"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        # Handle any size mismatch from odd dimensions
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.se(x)
        return x


class LightweightUNet(nn.Module):
    """
    U-Net style segmentation model with a pretrained MobileNetV2 encoder
    and SE attention blocks in the decoder.
    """
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()

        # Load pretrained MobileNetV2, we only need its 'features' (encoder) part
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        features = mobilenet.features

        # MobileNetV2's input expects 3 channels (RGB), our MRI images are grayscale (1 channel)
        # Replace the first conv layer to accept 1 channel instead of 3
        original_first_layer = features[0][0]
        features[0][0] = nn.Conv2d(
            1, original_first_layer.out_channels,
            kernel_size=original_first_layer.kernel_size,
            stride=original_first_layer.stride,
            padding=original_first_layer.padding,
            bias=False
        )

        # Split MobileNetV2 into stages to extract skip connections at different resolutions
        self.enc_stage1 = features[0:2]    # output: 16 channels,  128x128
        self.enc_stage2 = features[2:4]    # output: 24 channels,  64x64
        self.enc_stage3 = features[4:7]    # output: 32 channels,  32x32
        self.enc_stage4 = features[7:14]   # output: 96 channels,  16x16
        self.enc_stage5 = features[14:19]  # output: 1280 channels, 8x8 (bottleneck)

        # Decoder path: upsample back up, fusing skip connections at each stage
        self.dec4 = DecoderBlock(1280, 96, 256)
        self.dec3 = DecoderBlock(256, 32, 128)
        self.dec2 = DecoderBlock(128, 24, 64)
        self.dec1 = DecoderBlock(64, 16, 32)

        # Final upsample back to the original input resolution, then a 1x1 conv for the output mask
        self.final_upsample = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        s1 = self.enc_stage1(x)
        s2 = self.enc_stage2(s1)
        s3 = self.enc_stage3(s2)
        s4 = self.enc_stage4(s3)
        bottleneck = self.enc_stage5(s4)

        d4 = self.dec4(bottleneck, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        out = self.final_upsample(d1)
        out = self.final_conv(out)
        out = torch.sigmoid(out)  # binary segmentation: tumor vs background

        return out
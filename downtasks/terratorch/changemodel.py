import torch
import torch.nn as nn

from .backbone import (
    R50Backbone, ViTBackbone, SwinBackbone, PrithviBackbone, DoFABackbone,
    SatlasBackbone, SSL4EOBackbone, SatMAEBackbone, ScaleMAEBackbone,
    GFMBackbone, CROMABackbone, CrossScaleMAEBackbone, MaRSViTBackbone, MaRSBackbone
)


class CDBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels // 4, out_channels, kernel_size=1)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.deconv2(x)))
        x = self.relu3(self.norm3(self.conv3(x)))
        return x


class ChangeDecoder(nn.Module):
    def __init__(self, in_channels, upsample_times):
        super().__init__()
        self.block = CDBlock(in_channels * 2, 256)
        self.upsample_blocks = nn.ModuleList()
        in_ch = 256
        for _ in range(upsample_times):
            out_ch = max(in_ch // 2, 32)
            self.upsample_blocks.append(DecoderBlock(in_ch, out_ch))
            in_ch = out_ch
        self.final = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, feat1, feat2):
        x = torch.cat([feat1, feat2], dim=1)
        x = self.block(x)
        for up in self.upsample_blocks:
            x = up(x)
        x = self.final(x)
        return x


class ChangeDetectionModel(nn.Module):
    def __init__(self, backbone_class, in_channels=3, feat_dim=None, upsample_times=3):
        super().__init__()
        self.backbone = backbone_class(in_channels)
        self.decoder = ChangeDecoder(in_channels=feat_dim, upsample_times=upsample_times)

    def forward(self, img1, img2):
        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2)

        if isinstance(feat1, list):
            feat1 = feat1[-1]
            feat2 = feat2[-1]

        if feat1.dim() == 3:
            B, N, C = feat1.shape
            H = W = int(N ** 0.5)
            feat1 = feat1.permute(0, 2, 1).reshape(B, C, H, W)
            feat2 = feat2.permute(0, 2, 1).reshape(B, C, H, W)

        return self.decoder(feat1, feat2)


change_model_dict = {
    'r50_change': lambda: ChangeDetectionModel(R50Backbone, feat_dim=2048, upsample_times=3),
    'vit_change': lambda: ChangeDetectionModel(ViTBackbone, feat_dim=768, upsample_times=4),
    'swin_change': lambda: ChangeDetectionModel(SwinBackbone, feat_dim=1024, upsample_times=5),
    'prithvi_change': lambda: ChangeDetectionModel(PrithviBackbone, feat_dim=1024, upsample_times=4),
    'dofa_change': lambda: ChangeDetectionModel(DoFABackbone, feat_dim=768, upsample_times=4),
    'satlas_change': lambda: ChangeDetectionModel(SatlasBackbone, feat_dim=1024, upsample_times=5),
    'ssl4eo_change': lambda: ChangeDetectionModel(SSL4EOBackbone, feat_dim=2048, upsample_times=5),
    'satmae_change': lambda: ChangeDetectionModel(SatMAEBackbone, feat_dim=1024, upsample_times=4),
    'scalemae_change': lambda: ChangeDetectionModel(ScaleMAEBackbone, feat_dim=1024, upsample_times=4),
    'gfm_change': lambda: ChangeDetectionModel(GFMBackbone, feat_dim=1024, upsample_times=5),
    'croma_change': lambda: ChangeDetectionModel(CROMABackbone, feat_dim=768, upsample_times=3),
    'crossmae_change': lambda: ChangeDetectionModel(CrossScaleMAEBackbone, feat_dim=768, upsample_times=4),
    'marsvit_change': lambda: ChangeDetectionModel(MaRSViTBackbone, feat_dim=768, upsample_times=4),
    'mars_change': lambda: ChangeDetectionModel(MaRSBackbone, feat_dim=1024, upsample_times=5),
    'mars_large_change': lambda: ChangeDetectionModel(MaRSBackbone, feat_dim=1536, upsample_times=5),
}

if __name__ == '__main__':
    model = change_model_dict['crossmae_change']()
    img1 = torch.randn(2, 3, 512, 512)
    img2 = torch.randn(2, 3, 512, 512)
    out = model(img1, img2)
    print(out.shape)  # 应输出 [2, 1, 512, 512]

import torch
from torch import nn
import torch.nn.functional as F
from .backbone import MaRSBackbone, MaRSViTBackbone


def build_dual_backbone_model(backbone_class, feat_dim, up=5, rgb_in=3, sar_in=1):
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

    class DualBackboneModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.rgb_backbone = backbone_class(in_channels=rgb_in)
            self.sar_backbone = backbone_class(in_channels=sar_in)

            # 构建多层解码器（默认上采样到 512×512）
            if feat_dim == 2048:
                self.decoder4 = DecoderBlock(2048, 1024)
                self.decoder3 = DecoderBlock(1024, 512)
                self.decoder2 = DecoderBlock(512, 256)
                self.decoder1 = DecoderBlock(256, 64)
            elif feat_dim == 1536:
                self.decoder4 = DecoderBlock(1536, 768)
                self.decoder3 = DecoderBlock(768, 384)
                self.decoder2 = DecoderBlock(384, 192)
                self.decoder1 = DecoderBlock(192, 64)
            elif feat_dim == 1024:
                self.decoder4 = DecoderBlock(1024, 512)
                self.decoder3 = DecoderBlock(512, 256)
                self.decoder2 = DecoderBlock(256, 128)
                self.decoder1 = DecoderBlock(128, 64)
                
            elif feat_dim == 768:
                self.decoder4 = DecoderBlock(768, 384)
                self.decoder3 = DecoderBlock(384, 192)
                self.decoder2 = DecoderBlock(192, 96)
                self.decoder1 = DecoderBlock(96, 64)
            else:
                raise ValueError(f"Unsupported feat_dim={feat_dim}")
            if up == 4:
                self.final_deconv = nn.Conv2d(64, 32, kernel_size=3, padding=1) # 仅用于large版本的 vit
            elif up == 5:
                self.final_deconv = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
            elif up == 3:
                self.final_deconv = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1)
            self.final_relu = nn.ReLU(inplace=True)
            self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

        def forward(self, x):
            rgb = x[:, :3, :, :]
            sar = x[:, 3:, :, :]
            feat_rgb = self.rgb_backbone(rgb)
            feat_sar = self.sar_backbone(sar)
            # swin 输出的是多尺度特征列表
            if isinstance(feat_rgb, list):
                feat_rgb = feat_rgb[-1]
                feat_sar = feat_sar[-1] 
            # dofa 输出 torch.Size([2, 196, 768])，转为 torch.Size([2, 768, 14，14])
            if len(feat_rgb.shape) == 3:
                B, N, C = feat_rgb.shape
                H = W = int(N ** 0.5)
                feat_rgb = feat_rgb.permute(0, 2, 1).reshape(B, C, H, W)
                feat_sar = feat_sar.permute(0, 2, 1).reshape(B, C, H, W) 
            feat = feat_rgb + feat_sar  # shape: [B, C, 14, 14] or similar
            # print(f"Output shape: {feat.shape}")  # Debugging output
            x = self.decoder4(feat)
            x = self.decoder3(x)
            x = self.decoder2(x)
            x = self.decoder1(x)
            x = self.final_deconv(x)
            x = self.final_relu(x)
            x = self.final_conv(x)
            # print(f"Output shape: {x.shape}")  # Debugging output
            return x

    return DualBackboneModel

# swin_base 输出维度（固定）
# MaRSFCN = build_dual_backbone_model(MaRSBackbone, feat_dim=1024,up=5)
# swin_large 输出维度（固定）
MaRSFCN = build_dual_backbone_model(MaRSBackbone, feat_dim=1536,up=5)

MaRSViTFCN = build_dual_backbone_model(MaRSViTBackbone, feat_dim=768, up=4)
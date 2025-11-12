import torch
from torch import nn
import torch.nn.functional as F
from .backbone import MaRSBackbone


# -------------------- Activation --------------------
def nonlinearity(x):
    return torch.relu(x)


# -------------------- Decoder Block --------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(
            in_channels // 4, in_channels // 4,
            kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, kernel_size=1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.deconv2(x)))
        x = self.relu3(self.norm3(self.conv3(x)))
        return x


## 使用于siwnv2.0版本的MaRSFCN
class MaRSFCN(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super(MaRSFCN, self).__init__()
        self.encoder = MaRSBackbone(in_channels=in_channels)

        # swin_base 输出维度（固定）
        # filters = [128, 256, 512, 1024]
        # swin_large 输出维度（固定）
        filters = [192, 384, 768, 1536]
        # 多尺度解码器
        self.decoder4 = DecoderBlock(filters[3], filters[2])  # 7→14
        self.decoder3 = DecoderBlock(filters[2], filters[1])  # 14→28
        self.decoder2 = DecoderBlock(filters[1], filters[0])  # 28→56
        self.decoder1 = DecoderBlock(filters[0], filters[0])  # 56→112

        # 最终上采样模块（输出 224×224）
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2, padding=1, output_padding=1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 1)  # 最终分割类别数输出

    def forward(self, x):
        e1, e2, e3, e4 = self.encoder(x)  # 分别是 56×56、28×28、14×14、7×7

        d4 = self.decoder4(e4)# + e3       # → 14×14
        d3 = self.decoder3(d4)# + e2       # → 28×28
        d2 = self.decoder2(d3)# + e1       # → 56×56
        d1 = self.decoder1(d2)            # → 112×112

        out = self.finaldeconv1(d1)       # → 224×224
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)        # → num_classes × 224 × 224

        return out
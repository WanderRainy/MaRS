import torch
from torch import nn
from torchvision.models.segmentation import fcn_resnet50
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models import swin_transformer
from torchvision.models.swin_transformer import swin_b, Swin_B_Weights
import timm


import sys # 作为主函数测试时使用
sys.path.append('/mnt/data/MaRS/down_tasks/terratorch')  # 添加utils目录到路径中

from model_factory.utils.use_croma import PretrainedCROMA


class R50Backbone(nn.Module):
    def __init__(self, in_channels=3):  # 支持1或3通道
        super().__init__()
        assert in_channels in [1, 3], "Only in_channels=1 or 3 are supported"

        # 加载带主干预训练权重的 fcn_resnet50
        self.model = fcn_resnet50(weights_backbone=ResNet50_Weights.IMAGENET1K_V1)

        if in_channels != 3:
            old_conv = self.model.backbone.conv1
            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            # 权重初始化：复制或者平均
            with torch.no_grad():
                if in_channels == 1:
                    # 将原3通道权重在通道维度上平均为1通道
                    new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
                # 如果你想扩展为其他通道数，例如in_channels=5，可改为repeat或截断拼接
            self.model.backbone.conv1 = new_conv
        # 如果 in_channels == 3，就不修改，使用预训练原始网络

    def forward(self, x): # x shape: ([2, 3, 512, 512]) or ([2, 1, 512, 512])
        return self.model.backbone(x)['out'] # output shape: ([2, 2048, 64, 64])


class ViTBackbone(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        vit_name="vit_base_patch16_224"
        self.vit = timm.create_model(
            vit_name,
            pretrained=True,
            in_chans=in_channels,
            features_only=True,
            out_indices=[-1],
            img_size=512,
            cache_dir='/mnt/data/MaRS/down_tasks/terratorch/model_factory/pretrain_weights',
        )
    def forward(self, x): # x shape: ([2, 3, 512, 512]) or ([2, 1, 512, 512])
        return self.vit(x)[0]  # Return only the last layer features shape: ([2, 768, 28, 28])



class SwinBackbone(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        """
        Args:
            model_name (str): Swin 模型名称，如 'swin_base_patch4_window7_224'
            in_channels (int): 输入通道数（支持 1 或 3）
            pretrained (bool): 是否使用预训练模型
        """
        super().__init__()
        model_name='swin_base_patch4_window7_224'
        assert in_channels in [1, 3], "Only in_channels=1 or 3 supported."

        self.backbone = timm.create_model(
            'swinv2_base_window8_256',
            pretrained=True,
            features_only=True,
            in_chans=in_channels,
            img_size=512,
            cache_dir='/mnt/data/MaRS/models/pretrained_weights'
        )
    def forward(self, x): # x shape: ([2, 3, 512, 512]) or ([2, 1, 512, 512])
        """
        Args:
            x (Tensor): shape [B, C, H, W]
        Returns:
            List[Tensor]: 多尺度特征列表
            out1[0].shape
            torch.Size([2, 128, 128, 128])
            out1[1].shape
            torch.Size([2, 256, 56, 56])
            out1[2].shape
            torch.Size([2, 512, 25, 28])
            out1[3].shape
            torch.Size([2, 1024, 14, 14])
        """
        features = self.backbone(x)
        features = [f.permute(0, 3, 1, 2).contiguous() for f in features]  # 转成 NCHW
        return features


class DoFABackbone(nn.Module):# 需要修改dofa源码，源代码中只实现了分类的特征 
    def __init__(self, in_channels=3):
        super().__init__()
        assert in_channels in [1, 3], "Only support in_channels = 1 or 3"
        self.in_channels = in_channels
        # 加载 DOFA 模型
        self.model = torch.hub.load(
            '/mnt/data/MaRS/down_tasks/terratorch/model_factory/utils/zhu-xlab_DOFA_master',
            'vit_base_dofa',  # The entry point defined in hubconf.py
            pretrained=False,
            source='local'
        )
        self.model.load_state_dict(torch.load('/mnt/data/MaRS/down_tasks/terratorch/model_factory/pretrain_weights/DOFA_ViT_base_e100.pth'), strict=False)
    
    def forward(self, x):# x shape: ([2, 3, 224, 224]) or ([2, 1, 224, 224])
        if self.in_channels == 1:
            wavelengths = [0.64]  # RED only
        else:
            wavelengths = [0.48, 0.56, 0.64]  # B, G, R

        return self.model.forward_features(x, wavelengths)[:, 1:, :] # torch.Size([2, 196, 768])


class SatMAEBackbone(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        self.vit = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            in_chans=in_channels,
            img_size=512,
            features_only=True,
            out_indices=[-1],
        )
        self.vit.load_state_dict(torch.load('/mnt/data/MaRS/down_tasks/terratorch/model_factory/pretrain_weights/checkpoint_ViT-L_pretrain_fmow_rgb.pth', weights_only=False), strict=False)

    def forward(self, x): # x shape: ([2, 3, 512, 512]) or ([2, 1, 512, 512])
        return self.vit(x)[0]  # ([2, 1024, 28, 28])


class ScaleMAEBackbone(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        vit_name="vit_large_patch16_224"
        pretrained=False
        self.vit = timm.create_model(
            vit_name,
            pretrained=pretrained,
            in_chans=in_channels,
            img_size=512,
            features_only=True,
            out_indices=[-1]
        )
        self.vit.load_state_dict(torch.load('/mnt/data/MaRS/down_tasks/terratorch/model_factory/pretrain_weights/scalemae-vitlarge-800.pth', weights_only=False), strict=False)

    def forward(self, x): # x shape: ([2, 3, 224, 224]) or ([2, 1, 224, 224])
        return self.vit(x)[0]  # ([2, 1024, 14, 14])


class GFMBackbone(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        assert in_channels in [1, 3], "Only in_channels = 1 or 3 are supported"

        # 加载 Swin-B 模型
        model = swin_b(weights=None)  # 不加载 torchvision 的默认权重
        patch_embed = model.features[0]  # patch embedding 是 features[0]
        conv = patch_embed[0]  # Conv2d(3, 128, 4, 4)

        # 替换为支持 in_channels 的版本
        if in_channels != 3:
            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                bias=conv.bias is not None
            )
            with torch.no_grad():
                new_conv.weight[:] = conv.weight.mean(dim=1, keepdim=True)
            patch_embed[0] = new_conv  # 替换掉原始 conv 层

        self.backbone = model.features
        self.backbone.load_state_dict(torch.load('/mnt/data/MaRS/down_tasks/terratorch/model_factory/pretrain_weights/gfm.pth', weights_only=False), strict=False)

    def forward(self, x): # x shape: ([2, 3, 512, 512]) or ([2, 1, 512, 512])
        x = self.backbone(x)
        return x.permute(0, 3, 1, 2)  # 转换为 ([2, 1024, 16, 16])

class CROMABackbone(nn.Module):
    def __init__(self, in_channels=3):
        '''
        https://github.com/antofuller/CROMA/tree/main'''
        super().__init__()
        self.in_channels = in_channels
        if in_channels == 3:
            self.backbone = PretrainedCROMA(pretrained_path='/mnt/data/MaRS/down_tasks/terratorch/model_factory/pretrain_weights/CROMA_base.pt', size='base', modality='optical', image_resolution=512, optical_channels=3)
        else:
            self.backbone = PretrainedCROMA(pretrained_path='/mnt/data/MaRS/down_tasks/terratorch/model_factory/pretrain_weights/CROMA_base.pt', size='base', modality='SAR', image_resolution=512, sar_channels=1)
    def forward(self, x): # x shape: ([2, 3, 512, 512]) or ([2, 1, 512, 512])
        if self.in_channels == 1:
            return self.backbone(SAR_images=x)['SAR_encodings'] # ([2, 4096, 768]) # 64*64
        else:
            return self.backbone(optical_images=x)['optical_encodings'] # ([2, 4096, 768])


class CrossScaleMAEBackbone(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        vit_name="vit_base_patch16_224"
        pretrained=False
        self.vit = timm.create_model(
            vit_name,
            pretrained=pretrained,
            in_chans=in_channels,
            img_size=512,
            features_only=True,
            out_indices=[-1]
        )
        self.vit.load_state_dict(torch.load('/mnt/data/MaRS/down_tasks/terratorch/model_factory/pretrain_weights/cross_scale_mae_base_pretrain.pth', weights_only=False), strict=False)

    def forward(self, x): # x shape: ([2, 3, 512, 512]) or ([2, 1, 512, 512])
        return self.vit(x)[0] # ([2, 1024, 28, 28])

################################################python 3.11, torch 2.6.0, terratorch 1.0 ##################################################################
from terratorch import BACKBONE_REGISTRY
from terratorch.models import EncoderDecoderFactory
from terratorch.datasets import HLSBands


class PrithviBackbone(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        assert in_channels in [1, 3], "Only support in_channels = 1 or 3"
        
        # 加载权重路径
        backbone_ckpt_path = '/mnt/data/MaRS/down_tasks/terratorch/model_factory/pretrain_weights/Prithvi_EO_V2_300M.pt'
        
        # 根据通道数选择输入波段
        if in_channels == 3:
            input_bands = [HLSBands.RED, HLSBands.GREEN, HLSBands.BLUE]
        else:  # in_channels == 1
            input_bands = [HLSBands.RED]

        # 构建 Prithvi 模型
        self.backbone = BACKBONE_REGISTRY.build(
            "prithvi_eo_v2_300",
            pretrained=True,
            ckpt_path=backbone_ckpt_path,
            bands=input_bands
        )

    def forward(self, x): # x shape: ([2, 3, 512, 512]) or ([2, 1, 512, 512])
        # backbone 返回为 12*  of (B, 1025, 768)
        return self.backbone(x)[-1][:, 1:, :] # ([2, 1024, 768])------>(B, 32*32, C)
    
class SatlasBackbone(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        backbone_ckpt_path='/mnt/data/MaRS/down_tasks/terratorch/model_factory/pretrain_weights/aerial_swinb_si-e4169eb1.pth'

        assert in_channels in [1, 3], "Only in_channels = 1 or 3 supported"

        if in_channels == 3:
            input_bands = [HLSBands.RED, HLSBands.GREEN, HLSBands.BLUE]
        else:
            input_bands = [HLSBands.RED]

        self.backbone = BACKBONE_REGISTRY.build(
            "satlas_swin_b_naip_si_rgb",
            pretrained=False,
            model_bands=input_bands
        )
        self.backbone.load_state_dict(torch.load(backbone_ckpt_path, weights_only=False), strict=False)

    def forward(self, x): # x shape: ([2, 3, 512, 512]) or ([2, 1, 512, 512])
        return self.backbone(x)[0].permute(0, 3, 1, 2) #torch.Size([2, 1024, 16, 16])


class SSL4EOBackbone(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        backbone_ckpt_path='/mnt/data/MaRS/down_tasks/terratorch/model_factory/pretrain_weights/resnet50_sentinel2_all_dino-d6c330e9.pth'
        assert in_channels in [1, 3], "Only in_channels = 1 or 3 supported"

        if in_channels == 3:
            input_bands = [HLSBands.RED, HLSBands.GREEN, HLSBands.BLUE]
        else:
            input_bands = [HLSBands.RED]

        self.backbone = BACKBONE_REGISTRY.build(
            "ssl4eos12_resnet50_sentinel2_all_dino",
            pretrained=False,
            model_bands=input_bands
        )
        self.backbone.load_state_dict(torch.load(backbone_ckpt_path), strict=False)

    def forward(self, x): # x shape: ([2, 3, 512, 512]) or ([2, 1, 512, 512])
        return self.backbone(x)[0]#.permute(0, 3, 1, 2) #torch.Size([2, 2048, 16, 16])


class MaRSViTBackbone(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        pretrained=False

        vit_name="vit_base_patch16_224"
        if in_channels == 3:
            self.vit = timm.create_model(
                vit_name,
                pretrained=pretrained,
                in_chans=in_channels,
                features_only=True,
                out_indices=[-1],
                img_size=512,
                checkpoint_path='/mnt/data/MaRS/down_tasks/terratorch/model_factory/mars_weights/vit_rgb_encoder_only.pth',
            )
        elif in_channels == 1:
            self.vit = timm.create_model(
                vit_name,
                pretrained=pretrained,
                in_chans=in_channels,
                features_only=True,
                out_indices=[-1],
                img_size=512,
                checkpoint_path='/mnt/data/MaRS/down_tasks/terratorch/model_factory/mars_weights/vit_sar_encoder_only.pth',
            )
        
    def forward(self, x): # x shape: ([2, 3, 512, 512]) or ([2, 1, 512, 512])
        return self.vit(x)[0]  # Return only the last layer features shape: ([2, 768, 28, 28])


class MaRSBackbone(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        """
        Args:
            model_name (str): Swin 模型名称，如 'swin_base_patch4_window7_224'
            in_channels (int): 输入通道数（支持 1 或 3）
            pretrained (bool): 是否使用预训练模型
        """
        super().__init__()
        # swinv2
        if in_channels == 3:
            self.backbone = timm.create_model(
                'base:swinv2_base_window8_256', # large:swinv2_large_window12to16_192to256, base:swinv2_base_window8_256
                pretrained=False,
                features_only=True,
                in_chans=in_channels,
                img_size=512,
                checkpoint_path='/mnt/data/MaRS/down_tasks/terratorch/model_factory/mars_weights/simmim_swinb_attn_cross/swin_rgb_encoder_only.pth'
            )
        elif in_channels == 1:
            # 单通道输入，加载单通道预训练权重
            self.backbone = timm.create_model(
                'base:swinv2_base_window8_256', # large:swinv2_large_window12to16_192to256, base:swinv2_base_window8_256
                pretrained=False,
                features_only=True,
                in_chans=in_channels,
                img_size=512,
                checkpoint_path="/mnt/data/MaRS/down_tasks/terratorch/model_factory/mars_weights/simmim_swinb_attn_cross/swin_sar_encoder_only.pth"
            )
    def forward(self, x): # x shape: ([2, 3, 512, 512]) or ([2, 1, 512, 512])
        """
        Args:
            x (Tensor): shape [B, C, H, W]
        Returns:
            List[Tensor]: 多尺度特征列表
            out1[0].shape
            torch.Size([2, 128, 128, 128])
            out1[1].shape
            torch.Size([2, 256, 56, 56])
            out1[2].shape
            torch.Size([2, 512, 25, 28])
            out1[3].shape
            torch.Size([2, 1024, 14, 14])
        """
        features = self.backbone(x)
        features = [f.permute(0, 3, 1, 2).contiguous() for f in features]  # 转成 NCHW
        return features
# --------------------- 主函数测试 ---------------------
def main():
    # 测试单通道输入
    print("Testing 1-channel input...")
    model_1ch = MaRSBackbone(in_channels=1)
    x1 = torch.randn(2, 1, 512, 512)  # batch=2, 1通道
    out1 = model_1ch(x1)
    print("1-channel input output shape:", out1.shape)

    # 测试三通道输入
    # print("Testing 3-channel input...")
    # model_3ch = SSL4EOBackbone(in_channels=3)
    # x3 = torch.randn(2, 3, 512, 512)  # batch=2, 3通道
    # out3 = model_3ch(x3)
    # print("3-channel input output shape:", out3[-1].shape)

if __name__ == "__main__":
    main()
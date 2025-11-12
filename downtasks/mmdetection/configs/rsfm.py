# backbones.py
import torch
from torch import nn
from torchvision.models.segmentation import fcn_resnet50
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models import swin_transformer
from torchvision.models.swin_transformer import swin_b, Swin_B_Weights
import timm

from mmdet.registry import MODELS

from .use_croma import PretrainedCROMA

@MODELS.register_module()
class ViTBackbone(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        pretrained=True

        vit_name="vit_base_patch16_224"
        self.vit = timm.create_model(
            vit_name,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
            out_indices=[-1],
            img_size=512,
            cache_dir='/compared_FM/terratorch/pretrain_weights',
        )
        ## large版本，用来验证参数性能影响-建筑物高度估计
        # self.vit = timm.create_model(
        #     'vit_large_patch16_224',
        #     pretrained=True,
        #     in_chans=in_channels,
        #     features_only=True,
        #     out_indices=[-1],
        #     img_size=512,
        #     cache_dir='/compared_FM/terratorch/pretrain_weights',
        # )
        
        # dinov2 版本，在道路提取上 效果不佳
        # self.vit = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=False,
        #                              in_chans=in_channels,features_only=True,
        #                              out_indices=[-1],)
        # ckpt = torch.load("/compared_FM/terratorch/pretrain_weights/vit_base_patch14_dinov2.pth", map_location='cpu')
        # self.vit.load_state_dict(ckpt, strict=False)
    def forward(self, x): # x shape: ([2, 3, 512, 512]) or ([2, 1, 512, 512])
        return [self.vit(x)[0]]  # Return only the last layer features shape: ([2, 768, 28, 28])

@MODELS.register_module()
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
        assert in_channels in [1, 3, 4], "Only in_channels=1 or 3 supported."

        # 加载 timm 模型并启用多尺度特征
        # self.backbone = timm.create_model(
        #     model_name,
        #     pretrained=True,
        #     features_only=True,
        #     in_chans=in_channels,
        #     img_size=512,
        #     cache_dir='/compared_FM/terratorch/pretrain_weights'
        # )
        # swinv2
        self.backbone = timm.create_model(
            'swinv2_base_window8_256',
            pretrained=True,
            features_only=True,
            in_chans=in_channels,
            img_size=512,
            cache_dir='/models/pretrained_weights'
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


@MODELS.register_module()
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
        self.vit.load_state_dict(torch.load('/compared_FM/terratorch/pretrain_weights/checkpoint_ViT-L_pretrain_fmow_rgb.pth', weights_only=False), strict=False)

    def forward(self, x): # x shape: ([2, 3, 512, 512]) or ([2, 1, 512, 512])
        return [self.vit(x)[0]]  # ([2, 1024, 28, 28])


@MODELS.register_module()
class DoFABackbone(nn.Module):# 需要修改dofa源码，源代码中只实现了分类的特征 "/home/yry22/.cache/torch/hub/zhu-xlab_DOFA_master/dofa_v1.py"
    def __init__(self, in_channels=3):
        super().__init__()
        assert in_channels in [1, 3, 4], "Only support in_channels = 1 or 3, 4"
        self.in_channels = in_channels
        # 加载 DOFA 模型
        self.model = torch.hub.load(
            '/compared_FM/terratorch/model_factory/utils/zhu-xlab_DOFA_master',
            'vit_base_dofa',  # The entry point defined in hubconf.py
            pretrained=False,
            source='local'
        )
        self.model.load_state_dict(torch.load('/compared_FM/terratorch/pretrain_weights/DOFA_ViT_base_e100.pth'), strict=False)
    
    def forward(self, x): # x shape: ([2, 3, 224, 224]) or ([2, 1, 224, 224])
        if self.in_channels == 1:
            wavelengths = [0.64]  # RED only
        elif self.in_channels == 3:
            wavelengths = [0.48, 0.56, 0.64]  # B, G, R
        elif self.in_channels == 4:
            wavelengths = [0.48, 0.56, 0.64, 0.64]
        outs = []
        out = self.model.forward_features(x, wavelengths)[:, 1:, :]
        B, N, C = out.shape
        H = W = int(N ** 0.5)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        outs.append(out)
        return outs # torch.Size([2, 768, 14, 14])


@MODELS.register_module()
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
        self.vit.load_state_dict(torch.load('/compared_FM/terratorch/pretrain_weights/scalemae-vitlarge-800.pth', weights_only=False), strict=False)

    def forward(self, x): # x shape: ([2, 3, 512, 512]) or ([2, 1, 512, 512])
        return [self.vit(x)[0]]  # ([2, 1024, 28, 28])


@MODELS.register_module()
class GFMBackbone(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        assert in_channels in [1, 3, 4], "Only in_channels = 1, 3, or 4 are supported"

        # 加载 Swin-B 模型（不加载权重）
        model = swin_b(weights=None)
        patch_embed = model.features[0]  # patch embedding 层
        conv = patch_embed[0]  # 原始 Conv2d(3, 128, kernel_size=4, stride=4)

        # 替换为新的 conv，以支持不同通道数
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
                if in_channels == 1:
                    # 灰度图，复制权重的平均值
                    new_conv.weight[:] = conv.weight.mean(dim=1, keepdim=True)
                elif in_channels == 4:
                    # 四通道，前三通道复制 RGB，第四通道初始化为均值
                    new_conv.weight[:, :3] = conv.weight
                    new_conv.weight[:, 3:] = conv.weight.mean(dim=1, keepdim=True)
            patch_embed[0] = new_conv  # 替换原始 Conv2d 层

        self.backbone = model.features
        self.backbone.load_state_dict(
            torch.load('/compared_FM/terratorch/pretrain_weights/gfm.pth', map_location='cpu'),
            strict=False
        )
    def forward(self, x): # x shape: ([2, 3, 512, 512]) or ([2, 1, 512, 512])
        x = self.backbone(x)
        return [x.permute(0, 3, 1, 2)]  # 转换为 ([2, 1024, 16, 16])


@MODELS.register_module()
class CROMABackbone(nn.Module):
    def __init__(self, in_channels=3):
        '''
        https://github.com/antofuller/CROMA/tree/main'''
        super().__init__()
        if in_channels == 3:
            self.backbone = PretrainedCROMA(pretrained_path='/compared_FM/terratorch/pretrain_weights/CROMA_base.pt', size='base', modality='optical', image_resolution=512, optical_channels=3)
        elif in_channels == 1:
            self.backbone = PretrainedCROMA(pretrained_path='/compared_FM/terratorch/pretrain_weights/CROMA_base.pt', size='base', modality='SAR', image_resolution=512, sar_channels=1)
        elif in_channels == 4:
            self.backbone = PretrainedCROMA(pretrained_path='/compared_FM/terratorch/pretrain_weights/CROMA_base.pt', size='base', modality='both', image_resolution=512, sar_channels=1, optical_channels=3)
    def forward(self, x):
        feat = self.backbone(optical_images=x[:,:3,:,:], SAR_images=x[:,3:,:,:])['joint_encodings']
        B, N, C = feat.shape
        H = W = int(N ** 0.5)
        feat = feat.permute(0, 2, 1).reshape(B, C, H, W)
        return [feat]

@MODELS.register_module()
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
        self.vit.load_state_dict(torch.load('/compared_FM/terratorch/pretrain_weights/cross_scale_mae_base_pretrain.pth', weights_only=False), strict=False)

    def forward(self, x): # x shape: ([2, 3, 512, 512]) or ([2, 1, 512, 512])
        return [self.vit(x)[0]] # ([2, 1024, 28, 28])


@MODELS.register_module()
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
        self.in_channels = in_channels
        if in_channels == 3:
            self.backbone = timm.create_model(
                'swinv2_base_window8_256',
                pretrained=False,
                features_only=True,
                in_chans=in_channels,
                img_size=512,
                checkpoint_path="/mnt/data/MaRS/down_tasks/terratorch/model_factory/mars_weights/simmim_swinb_attn_cross/swin_rgb_encoder_only.pth"
            )
        elif in_channels == 1:
            # 单通道输入，加载单通道预训练权重
            self.backbone = timm.create_model(
                'swinv2_base_window8_256',
                pretrained=False,
                features_only=True,
                in_chans=in_channels,
                img_size=512,
                checkpoint_path="/mnt/data/MaRS/down_tasks/terratorch/model_factory/mars_weights/simmim_swinb_attn_cross/swin_sar_encoder_only.pth"
            )
        elif in_channels == 4:
            # 四通道输入，加载四通道预训练权重
            self.backbone_rgb = timm.create_model(
                'swinv2_base_window8_256',
                pretrained=False,
                features_only=True,
                in_chans=3,
                img_size=512,
                checkpoint_path="/mnt/data/MaRS/down_tasks/terratorch/model_factory/mars_weights/simmim_swinb_attn_cross/swin_rgb_encoder_only.pth"
            )
            self.backbone_sar = timm.create_model(
                'swinv2_base_window8_256',
                pretrained=False,
                features_only=True,
                in_chans=1,
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
        if self.in_channels == 4:
            # 四通道输入，分别处理 RGB 和 SAR 通道
            features_rgb = self.backbone_rgb(x[:, :3, :, :])
            features_sar = self.backbone_sar(x[:, 3:, :, :])
            features = [f_rgb + f_sar for f_rgb, f_sar in zip(features_rgb, features_sar)]
        else:
            features = self.backbone(x)
        features = [f.permute(0, 3, 1, 2).contiguous() for f in features]  # 转成 NCHW
        return features
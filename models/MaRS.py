import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.layers.weight_init import trunc_normal_
from timm.layers.drop import DropPath
from timm.layers.mlp import Mlp
from models.clips import clip_contrastive, cross_contrastive


class MetaAttention(nn.Module):
    def __init__(self, dim, num_layers, num_heads, mlp_ratio=4.0, drop=0.0, aa_order=('modality', 'global')):
        super().__init__()
        self.aa_block_num = num_layers
        self.aa_order = aa_order

        self.modality_blocks = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, num_heads, batch_first=True),
                DropPath(drop) if drop > 0 else nn.Identity(),
                Mlp(dim, int(dim * mlp_ratio), drop=drop),
            ]) for _ in range(num_layers)
        ])
        self.global_blocks = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, num_heads, batch_first=True),
                DropPath(drop) if drop > 0 else nn.Identity(),
                Mlp(dim, int(dim * mlp_ratio), drop=drop),
            ]) for _ in range(num_layers)
        ])

    def forward(self, rgb_feat, sar_feat):
        B, C, H, W = rgb_feat.shape
        P = H * W
        S = P
        rgb = rgb_feat.flatten(2).transpose(1, 2)  # [B, S, C]
        sar = sar_feat.flatten(2).transpose(1, 2)
        tokens = torch.cat([rgb, sar], dim=1)  # [B, 2S, C]

        modality_idx = 0
        global_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "modality":
                    ln, attn, dp, mlp = self.modality_blocks[modality_idx]
                    t = ln(tokens)
                    rgb_tok = t[:, :S]
                    sar_tok = t[:, S:]
                    out_rgb, _ = attn(rgb_tok, rgb_tok, rgb_tok)
                    out_sar, _ = attn(sar_tok, sar_tok, sar_tok)
                    merged = torch.cat([out_rgb, out_sar], dim=1)
                    tokens = tokens + dp(mlp(merged))
                    modality_idx += 1
                    modality_inter = tokens
                elif attn_type == "global":
                    ln, attn, dp, mlp = self.global_blocks[global_idx]
                    t = ln(tokens)
                    out, _ = attn(t, t, t)
                    tokens = tokens + dp(mlp(out))
                    global_idx += 1
                    global_inter = tokens
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            output_list.append(torch.cat([modality_inter, global_inter], dim=-1))

        rgb_out = tokens[:, :S, :].transpose(1, 2).reshape(B, C, H, W)
        sar_out = tokens[:, S:, :].transpose(1, 2).reshape(B, C, H, W)
        return rgb_out, sar_out, output_list


class SwinTransformerForMaRS(nn.Module):
    def __init__(self, model_name='swinv2_base_window8_256', img_size=512, in_chans=3, pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            img_size=img_size,
            cache_dir='models/pretrained_weights'
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.encoder.embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(self, x, mask):
        x = self.encoder.patch_embed(x)
        B, H, W, C = x.shape
        mask_token = self.mask_token.view(1, 1, 1, C).expand(B, H, W, C)
        mask = mask.unsqueeze(-1)
        x = x * (~mask) + mask_token * mask
        for layer in self.encoder.layers:
            x = layer(x)
        x = self.encoder.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mask_token'}


class DualEncoderMaRS(nn.Module):
    def __init__(self, encoder_rgb, encoder_sar, teacher_encoder, encoder_stride=32, contrast_type='none'):
        super().__init__()
        self.encoder_rgb = encoder_rgb
        self.encoder_sar = encoder_sar
        self.teacher_encoder = teacher_encoder
        self.encoder_stride = encoder_stride

        dim_out = [int(encoder_rgb.encoder.embed_dim * 2 ** i) for i in range(encoder_rgb.encoder.num_layers)][-1]
        self.decoder_rgb = nn.Sequential(
            nn.Conv2d(dim_out, encoder_stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(encoder_stride),
        )
        self.decoder_sar = nn.Sequential(
            nn.Conv2d(dim_out, encoder_stride ** 2 * 1, kernel_size=1),
            nn.PixelShuffle(encoder_stride),
        )

        self.in_chans_rgb = encoder_rgb.encoder.patch_embed.proj.weight.shape[1]
        self.in_chans_sar = encoder_sar.encoder.patch_embed.proj.weight.shape[1]
        self.patch_size = encoder_rgb.encoder.patch_embed.patch_size

        self.contrast_type = contrast_type
        self.contrast_temp = nn.Parameter(torch.tensor(1.0))
        self.cos = nn.CosineSimilarity(dim=1)

        self.alternating_attn = MetaAttention(
            dim=dim_out, num_layers=2, num_heads=8, aa_order=('modality', 'global')
        )

    def forward(self, samples, mask_ratio):
        x, mask = samples[0], samples[1]
        rgb, sar = x[:, :3], x[:, 3:]

        z_rgb_raw = self.encoder_rgb(rgb, mask)
        z_sar_raw = self.encoder_sar(sar, mask)
        z_rgb, z_sar, _ = self.alternating_attn(z_rgb_raw, z_sar_raw)

        rec_rgb = self.decoder_rgb(z_rgb)
        rec_sar = self.decoder_sar(z_sar)

        ph, pw = self.patch_size if isinstance(self.patch_size, tuple) else (self.patch_size, self.patch_size)
        mask_up = mask.repeat_interleave(ph, dim=1).repeat_interleave(pw, dim=2).unsqueeze(1).contiguous()

        loss_rgb = (F.l1_loss(rgb, rec_rgb, reduction='none') * mask_up).sum() / (mask_up.sum() + 1e-5) / self.in_chans_rgb
        loss_sar = (F.l1_loss(sar, rec_sar, reduction='none') * mask_up).sum() / (mask_up.sum() + 1e-5) / self.in_chans_sar
        loss_recon = loss_rgb + loss_sar

        z_rgb_tokens = z_rgb.flatten(2).transpose(1, 2)
        z_sar_tokens = z_sar.flatten(2).transpose(1, 2)
        with torch.no_grad():
            z_teacher = self.teacher_encoder(rgb, mask)
        z_teacher_tokens = z_teacher.flatten(2).transpose(1, 2)
        loss_teacher = -self.cos(z_rgb_tokens, z_teacher_tokens).mean()

        if self.contrast_type == 'clip':
            loss_contrast = clip_contrastive(z_rgb_tokens, z_sar_tokens, t=self.contrast_temp)
        elif self.contrast_type == 'cross':
            loss_contrast = cross_contrastive(z_rgb_tokens, z_sar_tokens)
        else:
            loss_contrast = 0

        total_loss = loss_recon + loss_teacher + loss_contrast
        return total_loss, z_rgb_tokens, z_sar_tokens


def build_mars(model_type='base', contrast_type='clip'):
    if model_type == 'base':
        model_name = 'swinv2_base_window8_256'
    elif model_type == 'large':
        model_name = 'swinv2_large_window12to16_192to256'
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    encoder_rgb = SwinTransformerForMaRS(model_name, img_size=512, in_chans=3, pretrained=True)
    encoder_sar = SwinTransformerForMaRS(model_name, img_size=512, in_chans=1, pretrained=True)
    teacher_encoder = SwinTransformerForMaRS(model_name, img_size=512, in_chans=3, pretrained=True)
    for p in teacher_encoder.parameters():
        p.requires_grad = False

    return DualEncoderMaRS(
        encoder_rgb=encoder_rgb,
        encoder_sar=encoder_sar,
        teacher_encoder=teacher_encoder,
        encoder_stride=32,
        contrast_type=contrast_type
    )

import torch
import torch.nn as nn
import torch.nn.functional as F

def clip_contrastive(rgb_feats, sar_feats, t):
        # --------对比学习对齐两个模态
    # fea 2,50,768
    # without cls tokens
    rgb_fea = rgb_feats[:, 1:, :]  # 变为 [2, 49, 768]
    sar_fea = sar_feats[:, 1:, :]  # 变为 [2, 49, 768]

    rgb_fea = rgb_fea.reshape(-1, rgb_fea.shape[-1])  # 变为 [98, 768]
    sar_fea = sar_fea.reshape(-1, rgb_fea.shape[-1])  # 变为 [98, 768]

    # 3. L2 范数归一化
    rgb_fea = F.normalize(rgb_fea, p=2, dim=1)  # 每行进行 L2 归一化
    sar_fea = F.normalize(sar_fea, p=2, dim=1)  # 每行进行 L2 归一化
    logits = torch.matmul(rgb_fea, sar_fea.T)  # [98, 98], 两两之间的余弦相似度
    logits = logits * torch.exp(t)  # 可选：按照图片中乘以标量 t (假设 t=1)
    labels = torch.arange(logits.size(0)).to(logits.device)  # [0, 1, 2, ..., 97]
    # 分别计算两个方向的交叉熵损失
    loss_rgb = F.cross_entropy(logits, labels)  # vit_fea -> sar_fea
    loss_sar = F.cross_entropy(logits.T, labels)  # sar_fea -> vit_fea
    # 平均两个方向的 loss
    loss_clip = (loss_rgb + loss_sar) / 2

    return loss_clip


def cross_contrastive(rgb_feats, sar_feats, 
                     temp_patch=0.1, temp_global=0.05, temp_hier=0.07,
                     alpha=0.5, beta=0.3, gamma=0.2):
    """
    多层次跨模态对比损失函数
    Args:
        rgb_feats: RGB特征 [B, N, D]
        sar_feats: SAR特征 [B, N, D]
        temp_*: 各层级的温度系数
        alpha, beta, gamma: 损失权重
    Returns:
        contrastive_loss: 综合对比损失
    """
    
    
    # ==================== 特征预处理 ====================
    # 移除CLS token并归一化 (假设输入已处理)
    rgb_patches = F.normalize(rgb_feats, p=2, dim=-1)[:, 1:, :]  # [B, N, D]
    sar_patches = F.normalize(sar_feats, p=2, dim=-1)[:, 1:, :]
    B, N, D = rgb_patches.shape
    # ==================== 全局特征提取 ====================
    rgb_global = rgb_patches.mean(dim=1)  # [B, D]
    sar_global = sar_patches.mean(dim=1)
    
    # ==================== 核心对比计算 ====================
    # 1. Patch-to-Patch对比
    patch_sim = torch.einsum('bnd,jmd->bnjm', rgb_patches, sar_patches) / temp_patch
    patch_sim = patch_sim.reshape(B*N, B*N)  # [BN, BN]
    # 2. Image-to-Image对比
    global_sim = torch.mm(rgb_global, sar_global.T) / temp_global  # [B, B]
    
    # 3. 双向Hierarchical对比
    rgb_global_expanded = rgb_global.unsqueeze(1).repeat(1, N, 1)
    sar_global_expanded = sar_global.unsqueeze(1).repeat(1, N, 1)

    crossrgb_sim = torch.einsum('bnd,jmd->bnjm', rgb_patches, sar_global_expanded) / temp_hier 
    crossrgb_sim = crossrgb_sim.reshape(B*N, B*N)  # [BN, BN]
    crosssar_sim = torch.einsum('bnd,jmd->bnjm', sar_patches, rgb_global_expanded) / temp_hier 
    crosssar_sim = crosssar_sim.reshape(B*N, B*N)  # [BN, BN]

    # ==================== 损失计算 ====================
    device = rgb_feats.device
    
    # 1. Patch-level损失
    patch_labels = torch.arange(B*N, device=device)
    patch_loss = (F.cross_entropy(patch_sim, patch_labels) +
                 F.cross_entropy(patch_sim.T, patch_labels)) / 2
    
    # 2. Image-level损失
    img_labels = torch.arange(B, device=device)
    img_loss = (F.cross_entropy(global_sim, img_labels) +
                F.cross_entropy(global_sim.T, img_labels)) / 2
    
    # 3. Hierarchical损失
    hier_labels = torch.arange(B, device=device).repeat_interleave(N)  # [BN]

    hier_loss = (F.cross_entropy(crossrgb_sim, hier_labels) +
                 F.cross_entropy(crosssar_sim, hier_labels)) / 2
    
    # ==================== 加权融合 ====================
    total_loss = (alpha * patch_loss + 
                 beta * img_loss + 
                 gamma * hier_loss)
    
    return total_loss
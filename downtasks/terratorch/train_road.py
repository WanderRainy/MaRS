import argparse

import torch
torch.set_float32_matmul_precision('medium')  # 最佳实践：兼顾速度和精度
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torchmetrics import JaccardIndex
# from typing import Optional
from segmentation_models_pytorch.losses import DiceLoss
from pytorch_lightning.loggers import TensorBoardLogger
from dataset_factory import DPGlobeDataModule
# even though we don't use the import directly, we need it so that the models are available in the timm registry
# Model_Factory
# from model_factory import R50_FCN, ViTFCN, SwinFCN, PrithviFCN
from model_factory.segmodel import MaRSFCN
# 定义Lightning模型
class SegmentationModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        # if args.model == 'r50_fcn':
        #     # 使用ResNet50模型
        #     self.model = R50_FCN(num_classes=args.num_classes)
        # elif args.model == 'vit_fcn':
        #     # 使用ViT模型
        #     self.model = ViTFCN(num_classes=args.num_classes)
        # elif args.model == 'swin_fcn':
        #     # 使用ViT模型
        #     self.model = SwinFCN(num_classes=args.num_classes)
        # elif args.model == 'prithvi_fcn':
        #     # 使用Prithvi模型
        #     self.model = PrithviFCN(num_classes=args.num_classes)
        # elif args.model == 'dofa_fcn':
        #     # 使用DoFA模型
        #     self.model = DoFAFCN(num_classes=args.num_classes)
        # elif args.model == 'satlas_fcn':
        #     # 使用Satlas模型
        #     self.model = SatlasFCN(num_classes=args.num_classes)
        # elif args.model == 'ssl4eo_fcn':
        #     # 使用Satlas模型
        #     self.model = SSL4EOFCN(num_classes=args.num_classes)
        # elif args.model == 'satmae_fcn':
        #     # 使用Satlas模型
        #     self.model = SatMAEFCN(num_classes=args.num_classes)
        # elif args.model == 'scalemae_fcn':
        #     # 使用Satlas模型
        #     self.model = ScaleMAEFCN(num_classes=args.num_classes)
        # elif args.model == 'gfm_fcn':
        #     # 使用Satlas模型
        #     self.model = GFMFCN(num_classes=args.num_classes)
        # elif args.model == 'croma_fcn':
        #     # 使用CROMO模型
        #     self.model = CROMAFCN(num_classes=args.num_classes)
        # elif args.model == 'crossmae_fcn':
        #     # 使用CROMO模型
        #     self.model = CrossScaleMAEFCN(num_classes=args.num_classes)
        # elif args.model == 'mars_fcn':
            # 使用MaRS模型
        self.model =MaRSFCN(num_classes=args.num_classes)

        # 用于评估的指标
        self.train_iou = JaccardIndex(task="multiclass", num_classes=args.num_classes)
        self.val_iou = JaccardIndex(task="multiclass", num_classes=args.num_classes)
        
    def forward(self, x):
        out = self.model(x)
        return out
        # if isinstance(out, dict):
        #     return out
        # elif isinstance(out, torch.tensor):
        #     return out
        # else:  
        #     return out.output
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.args.epochs // 4, gamma=0.5)
        # print(self.hparams.keys())
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "epoch",  # 'epoch' 或 'step'
            "frequency": 1,       # 每 1 个 interval 更新一次
        },
    }
    
    def training_step(self, batch):
        images, masks = batch
        outputs = self(images)
        # 定义组合损失
        ce_loss = nn.CrossEntropyLoss()
        dice_loss = DiceLoss(mode='multiclass', from_logits=True)  # 多分类任务

        # 加权组合
        loss = 0.5 * ce_loss(outputs, masks) + 0.5 * dice_loss(outputs, masks)  
        # loss = F.cross_entropy(outputs, masks)
        
        # 计算IoU
        preds = torch.argmax(outputs, dim=1)
        self.train_iou.update(preds, masks)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", current_lr, prog_bar=True, logger=True)
        iou = self.train_iou.compute()
        self.log("train_iou", iou, prog_bar=True)
        self.train_iou.reset()
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, masks)
        
        # 计算IoU
        preds = torch.argmax(outputs, dim=1)
        self.val_iou.update(preds, masks)
        
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        iou = self.val_iou.compute()
        self.log("val_iou", iou, prog_bar=True)
        self.val_iou.reset()


# 训练函数
def train(args):
    # 初始化数据模块
    datamodule = DPGlobeDataModule(crop_size=args.crop_size, batch_size=args.batch_size, num_workers=args.num_workers)
    
    # 初始化模型
    model = SegmentationModel(args)

    # 定义日志记录器
    logger = TensorBoardLogger(
        save_dir="/mnt/data/MaRS/down_tasks/terratorch",  # 根目录
        name="experiments_road",        # 替代原来的 'lightning_logs'
        version=args.experiment_name                 # 指定版本号（可选）
    )
    # 设置回调函数
    checkpoint_callback = ModelCheckpoint(
        dirpath=logger.log_dir,
        monitor='val_iou',
        mode='max',
        save_top_k=1,
        filename='best-{epoch:02d}-{val_iou:.2f}'
    )
    
    # 初始化训练器
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=torch.cuda.device_count() if torch.cuda.is_available() else None,
        strategy=DDPStrategy(find_unused_parameters=True), # 要留，有些模型需要，有些不需要
        callbacks=[checkpoint_callback],
        log_every_n_steps=10
    )
    print('start training')
    # 开始训练
    trainer.fit(model, datamodule=datamodule)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Train a segmentation model based on lightning framework")
    
    # 通用参数
    parser.add_argument("--experiment_name", type=str, default="prithvi_deepglobe",
                        help="model+dataset")
    parser.add_argument("--model", type=str, default="prithvi_fcn",
                       help="模型名称 (r50_fcn/vit_fcn/swin_fcn/prithvi_fcn/dofa_fcn/satlas_fcn/ssl4eo_fcn/satmae_fcn/scalemae_fcn/gfm_fcn/croma_fcn/crossmae_fcn/mars_fcn)")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
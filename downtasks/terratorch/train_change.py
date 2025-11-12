import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torchmetrics import JaccardIndex
from segmentation_models_pytorch.losses import DiceLoss
from pytorch_lightning.loggers import TensorBoardLogger

from dataset_factory import WHUCDDataModule
from model_factory.changemodel import change_model_dict  # 假设你定义了变化检测模型集合


class ChangeModelPL(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.model = change_model_dict[args.model]()  # 从模型字典中调用
        
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(mode='binary', from_logits=True)

        self.train_iou = JaccardIndex(task="binary")
        self.val_iou = JaccardIndex(task="binary")

    def forward(self, img1, img2):
        return self.model(img1, img2)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.args.epochs // 4, gamma=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def training_step(self, batch, batch_idx):
        img, label, _ = batch
        img1, img2 = img[:, 0:3, :, :], img[:, 3:6, :, :]
        logits = self(img1, img2)
        loss = 0.5 * self.loss_fn(logits, label.unsqueeze(1).float()) + 0.5 * self.dice_loss(logits, label.unsqueeze(1).float())

        pred = (logits.sigmoid() > 0.5).int()
        self.train_iou.update(pred.squeeze(1), label.int())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        iou = self.train_iou.compute()
        self.log("train_iou", iou, prog_bar=True)
        self.train_iou.reset()
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)

    def validation_step(self, batch, batch_idx):
        img, label, _ = batch
        img1, img2 = img[:, 0:3, :, :], img[:, 3:6, :, :]
        logits = self(img1, img2)
        loss = self.loss_fn(logits, label.unsqueeze(1).float()) + self.dice_loss(logits, label.unsqueeze(1).float())
        
        pred = (logits.sigmoid() > 0.5).int()
        self.val_iou.update(pred.squeeze(1), label.int())
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        iou = self.val_iou.compute()
        self.log("val_iou", iou, prog_bar=True)
        self.val_iou.reset()


def train(args):
    datamodule = WHUCDDataModule(resize_size=args.crop_size, batch_size=args.batch_size, num_workers=args.num_workers)

    model = ChangeModelPL(args)

    logger = TensorBoardLogger(
        save_dir="./experiments_change",
        name=args.experiment_name
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=logger.log_dir,
        monitor='val_iou',
        mode='max',
        save_top_k=1,
        filename='best-{epoch:02d}-{val_iou:.2f}'
    )

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=torch.cuda.device_count(),
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=[checkpoint_callback],
        log_every_n_steps=10
    )

    trainer.fit(model, datamodule=datamodule)


def parse_args():
    parser = argparse.ArgumentParser(description="Train change detection model")
    parser.add_argument("--experiment_name", type=str, default="swin_change_whucd")
    parser.add_argument("--model", type=str, default="swin_change")  # 从 change_model_dict 中选择
    parser.add_argument("--num_classes", type=int, default=1)  # 二类
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

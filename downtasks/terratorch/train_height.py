import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from dataset_factory import DFC23Track2DataModule
from model_factory.heightmodel import MaRSFCN,MaRSViTFCN
import numpy as np


class Result:
    def __init__(self):
        self.reset()

    def reset(self):
        self.mse = 0
        self.rmse = 0
        self.mae = 0
        self.delta1 = 0
        self.delta2 = 0
        self.delta3 = 0

    def evaluate(self, output, target):
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        output[output <= 0] = 1e-5
        target[target <= 0] = 1e-5

        valid_mask = target > 0
        output = output[valid_mask]
        target = target[valid_mask]

        abs_diff = np.abs(output - target)
        self.mse = np.mean(abs_diff ** 2)
        self.rmse = np.sqrt(self.mse)
        self.mae = np.mean(abs_diff)

        max_ratio = np.maximum(output / target, target / output)
        self.delta1 = (max_ratio < 1.25).mean()
        self.delta2 = (max_ratio < 1.25 ** 2).mean()
        self.delta3 = (max_ratio < 1.25 ** 3).mean()


class HeightSegModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()

        model_map = {
            # 'r50_fcn': R50_FCN,
            # 'vit_fcn': ViTFCN,
            # 'swin_fcn': SwinFCN,
            # 'prithvi_fcn': PrithviFCN,
            # 'dofa_fcn': DoFAFCN,
            # 'satlas_fcn': SatlasFCN,
            # 'ssl4eo_fcn': SSL4EOFCN,
            # 'satmae_fcn': SatMAEFCN,
            # 'scalemae_fcn': ScaleMAEFCN,
            # 'gfm_fcn': GFMFCN,
            # 'croma_fcn': CROMAFCN,
            # 'crossmae_fcn': CrossScaleMAEFCN,
            'mars_fcn': MaRSFCN,
            'mars_vit_fcn': MaRSViTFCN
        }

        self.model = model_map[args.model]()
        self.loss_reg = nn.L1Loss()
        self.metric = Result()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.args.epochs // 4, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def shared_step(self, batch, stage):
        x, height = batch  # x: (B, 4, H, W), height: (B, H, W)
        out = self(x)
        reg_pred = out.squeeze(1)

        loss_reg = self.loss_reg(reg_pred, height)
        total_loss = loss_reg

        if stage == 'val':
            self.metric.evaluate(reg_pred, height)
            self.log("val_delta1", self.metric.delta1, prog_bar=True, sync_dist=True)
            self.log("val_delta2", self.metric.delta2, sync_dist=True)
            self.log("val_delta3", self.metric.delta3, sync_dist=True)

        self.log(f"{stage}_loss", total_loss, prog_bar=True, sync_dist=True)
        return total_loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, stage='train')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, stage='val')


def train(args):
    datamodule = DFC23Track2DataModule(batch_size=args.batch_size, crop_size=args.crop_size, num_workers=args.num_workers)
    model = HeightSegModel(args)

    logger = TensorBoardLogger(save_dir="/mnt/data/MaRS/down_tasks/terratorch", name="experiments_regression", version=args.experiment_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=logger.log_dir,
        monitor='val_delta1',
        mode='max',
        save_top_k=1,
        filename='best-{epoch:02d}-{val_delta1:.3f}'
    )

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=torch.cuda.device_count() if torch.cuda.is_available() else None,
        strategy=DDPStrategy(find_unused_parameters=True), # 要留，有些模型需要，有些不需要
        callbacks=[checkpoint_callback],
        log_every_n_steps=10
    )

    trainer.fit(model, datamodule=datamodule)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a building height regression model")
    parser.add_argument("--experiment_name", type=str, default="regression_building")
    parser.add_argument("--model", type=str, default="prithvi_fcn")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

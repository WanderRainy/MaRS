import torch
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from torchmetrics import Precision, Recall, F1Score, JaccardIndex
from tqdm import tqdm
import torch.nn.functional as F
from dataset_factory import WHUCDDataModule
from train_change import ChangeModelPL

class ChangeDetectionInferencer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 数据模块
        self.datamodule = WHUCDDataModule(
            resize_size=args.resize_size,
            batch_size=1,
            num_workers=args.num_workers
        )
        self.datamodule.setup(stage='test')

        # 加载模型
        self.model = self._load_model(args.ckpt_path)
        self.model.to(self.device)
        self.model.eval()

        # 评价指标
        self.precision = Precision(task='binary').to(self.device)
        self.recall = Recall(task='binary').to(self.device)
        self.f1 = F1Score(task='binary').to(self.device)
        self.iou = JaccardIndex(task='binary').to(self.device)

        # 输出目录
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pred_dir = self.output_dir / "test_pred"
        self.pred_dir.mkdir(exist_ok=True)

    def _load_model(self, ckpt_path):
        model = ChangeModelPL.load_from_checkpoint(ckpt_path)
        return model

    def _save_prediction(self, img1, img2, true_mask, pred_mask, filename):
        """
        保存一张可视化拼接图：输入A/B + GT + Pred
        """
        # 转numpy
        img1 = (img1.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        img2 = (img2.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        gt = (true_mask.squeeze().cpu().numpy()*255).astype(np.uint8)
        pred = (pred_mask.squeeze().cpu().numpy()*255).astype(np.uint8)

        # 拼成三通道灰度
        gt_rgb = np.stack([gt]*3, axis=-1)
        pred_rgb = np.stack([pred]*3, axis=-1)

        # 拼接: [img1 | img2 | gt | pred]
        concat = np.hstack([img1, img2, gt_rgb, pred_rgb])

        # 保存
        Image.fromarray(concat).save(self.pred_dir / filename)

    def _update_metrics(self, preds, targets):
        self.precision.update(preds, targets)
        self.recall.update(preds, targets)
        self.f1.update(preds, targets)
        self.iou.update(preds, targets)

    def run_inference(self):
        loader = self.datamodule.test_dataloader()

        with torch.no_grad():
            for batch in tqdm(loader, desc="Inference"):
                imgs, labels, filenames = batch
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                img1 = imgs[:, :3, :, :]
                img2 = imgs[:, 3:6, :, :]

                logits = self.model(img1, img2)
                preds = (logits.sigmoid() > 0.5).int()

                for i in range(preds.shape[0]):
                    self._save_prediction(
                        img1[i].cpu(),
                        img2[i].cpu(),
                        labels[i].cpu(),
                        preds[i].cpu(),
                        filenames[i]
                    )
                    self._update_metrics(preds[i].squeeze(0), labels[i])

        metrics = {
            "Precision": self.precision.compute().item(),
            "Recall": self.recall.compute().item(),
            "F1": self.f1.compute().item(),
            "IoU": self.iou.compute().item()
        }
        print("\nEvaluation Results:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        with open(self.output_dir / "evaluation_results.txt", "w") as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./predictions")
    parser.add_argument("--resize_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    inferencer = ChangeDetectionInferencer(args)
    inferencer.run_inference()

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from torchmetrics import Precision, Recall, F1Score, JaccardIndex
from tqdm import tqdm
import torch.nn.functional as F
from dataset_factory import DPGlobeDataModule
from train_road import SegmentationModel

class RoadSegmentationInferencer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化数据模块
        self.datamodule = DPGlobeDataModule(
            crop_size=args.crop_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        self.datamodule.setup(stage='test')

        # 加载模型
        self.model = self._load_model(args.ckpt_path)
        self.model.to(self.device)
        self.model.eval()

        # 初始化评价指标
        self.precision = Precision(task='binary', average='macro').to(self.device)
        self.recall = Recall(task='binary', average='macro').to(self.device)
        self.f1 = F1Score(task='binary', average='macro').to(self.device)
        self.iou = JaccardIndex(task='binary').to(self.device)

        # 创建输出目录
        self.output_dir = Path(args.output_dir)
        self.output_test_dir = self.output_dir / 'test_pred'
        self.output_test_dir.mkdir(parents=True, exist_ok=True)

    def _load_model(self, ckpt_path):
        model = SegmentationModel.load_from_checkpoint(checkpoint_path=ckpt_path)
        return model

    def _save_prediction(self, image, true_mask, pred_mask, filename):
        # 转换为numpy数组
        image = (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        true_mask = (true_mask.cpu().numpy() * 255).astype(np.uint8)
        pred_mask = (pred_mask.cpu().numpy() * 255).astype(np.uint8)

        # 创建对比图
        comparison = np.hstack([
            image,
            np.stack([true_mask]*3, axis=-1),
            np.stack([pred_mask]*3, axis=-1)
        ])

        # 使用原始文件名保存
        save_path = self.output_test_dir / filename
        Image.fromarray(comparison).save(save_path)

    def _update_metrics(self, preds, targets):
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)

        self.precision.update(preds_flat, targets_flat)
        self.recall.update(preds_flat, targets_flat)
        self.f1.update(preds_flat, targets_flat)
        self.iou.update(preds_flat, targets_flat)

    def _sliding_window_predict(self, image):
        crop_size = self.args.crop_size
        stride = crop_size

        _, H, W = image.shape
        C = self.model.hparams.num_classes if hasattr(self.model.hparams, 'num_classes') else 2

        prob_map = torch.zeros((C, H, W), device=self.device)
        count_map = torch.zeros((1, H, W), device=self.device)

        for y in range(0, H, stride):
            for x in range(0, W, stride):
                y1, y2 = y, min(y + crop_size, H)
                x1, x2 = x, min(x + crop_size, W)

                patch = image[:, y1:y2, x1:x2].unsqueeze(0)
                pad_bottom = crop_size - (y2 - y1)
                pad_right = crop_size - (x2 - x1)
                patch = F.pad(patch, (0, pad_right, 0, pad_bottom), mode='reflect')

                with torch.no_grad():
                    out_patch = self.model(patch)
                    out_patch = out_patch[:, :, :y2 - y1, :x2 - x1]

                prob_map[:, y1:y2, x1:x2] += out_patch[0]
                count_map[:, y1:y2, x1:x2] += 1

        prob_map = prob_map / count_map
        pred = torch.argmax(prob_map, dim=0)
        return pred

    def run_inference(self):
        dataloader = self.datamodule.test_dataloader()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Processing"):
                images, masks, filenames = batch
                for i in range(images.shape[0]):
                    image = images[i].to(self.device)
                    mask = masks[i].to(self.device)
                    filename = filenames[i]

                    pred = self._sliding_window_predict(image)

                    self._save_prediction(image, mask, pred, filename)
                    self._update_metrics(pred.unsqueeze(0), mask.unsqueeze(0))

        metrics = {
            "Precision": self.precision.compute().item(),
            "Recall": self.recall.compute().item(),
            "F1": self.f1.compute().item(),
            "IoU": self.iou.compute().item()
        }

        print("\nEvaluation Results:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        with open(self.output_dir / "evaluation_results.txt", 'w') as f:
            for name, value in metrics.items():
                f.write(f"{name}: {value:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="prithvi_fcn",
                       help="Model name")
    parser.add_argument("--ckpt_path", type=str, required=True,
                       help="Path to trained checkpoint")
    parser.add_argument("--output_dir", type=str, default="./predictions",
                       help="Directory to save predictions")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Inference batch size (set to 1 for sliding window)")
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loader workers")

    args = parser.parse_args()
    inferencer = RoadSegmentationInferencer(args)
    inferencer.run_inference()

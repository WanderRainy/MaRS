import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from PIL import Image
import torch.nn.functional as F

from dataset_factory import DFC23Track2DataModule
from train_height import HeightSegModel


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


class BuildingHeightInferencer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.datamodule = DFC23Track2DataModule(
            crop_size=args.crop_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        self.datamodule.setup(stage='test')

        self.model = HeightSegModel.load_from_checkpoint(checkpoint_path=args.ckpt_path)
        self.model.to(self.device)
        self.model.eval()

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pred_vis_dir = self.output_dir / 'test_pred'
        self.pred_vis_dir.mkdir(exist_ok=True)

        self.label_max = 183.17412
        self.metric = Result()

    def _save_prediction(self, pred_tensor, filename):
        pred = pred_tensor.squeeze().cpu().numpy() * self.label_max
        pred_img = (pred / self.label_max * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(pred_img).save(self.pred_vis_dir / filename)

    def _sliding_window_predict(self, image, patch_size=224, stride=224):
        _, H, W = image.shape
        prob_map = torch.zeros((1, H, W), device=self.device)
        count_map = torch.zeros((1, H, W), device=self.device)

        for y in range(0, H, stride):
            for x in range(0, W, stride):
                y1, y2 = y, min(y + patch_size, H)
                x1, x2 = x, min(x + patch_size, W)

                patch = image[:, y1:y2, x1:x2].unsqueeze(0)
                pad_bottom = patch_size - (y2 - y1)
                pad_right = patch_size - (x2 - x1)

                patch = F.pad(patch, (0, pad_right, 0, pad_bottom), mode='constant', value=0)


                with torch.no_grad():
                    pred_patch = self.model(patch)
                    pred_patch = pred_patch[:, :, :y2 - y1, :x2 - x1]

                prob_map[:, y1:y2, x1:x2] += pred_patch[0]
                count_map[:, y1:y2, x1:x2] += 1

        return (prob_map / count_map).squeeze(0)

    def run_inference(self):
        dataloader = self.datamodule.test_dataloader()

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(dataloader)):
                images = images.to(self.device)
                labels = labels.to(self.device)

                for i in range(images.size(0)):
                    image = images[i]
                    label = labels[i]

                    if self.args.crop_size < image.shape[1]:  # 启用滑窗
                        pred = self._sliding_window_predict(image, patch_size=self.args.crop_size, stride=self.args.crop_size)
                    else:  # 直接整图预测
                        pred = self.model(image.unsqueeze(0)).squeeze(1).squeeze(0)

                    self._save_prediction(pred, f"pred_{batch_idx}_{i}.png")
                    self.metric.evaluate(pred, label)

        print("\nEvaluation Results:")
        print(f"MAE: {self.metric.mae:.4f}")
        print(f"MSE: {self.metric.mse:.4f}")
        print(f"RMSE: {self.metric.rmse:.4f}")
        print(f"delta1: {self.metric.delta1:.4f}")
        print(f"delta2: {self.metric.delta2:.4f}")
        print(f"delta3: {self.metric.delta3:.4f}")

        with open(self.output_dir / "evaluation_results.txt", 'w') as f:
            f.write(f"MAE: {self.metric.mae:.4f}\n")
            f.write(f"MSE: {self.metric.mse:.4f}\n")
            f.write(f"RMSE: {self.metric.rmse:.4f}\n")
            f.write(f"delta1: {self.metric.delta1:.4f}\n")
            f.write(f"delta2: {self.metric.delta2:.4f}\n")
            f.write(f"delta3: {self.metric.delta3:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./predictions")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    inferencer = BuildingHeightInferencer(args)
    inferencer.run_inference()

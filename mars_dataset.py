# References: MAE, and SIMMIM 
# Author: WanderRainY
# Modified: 2025-10
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile as tiff
import torchvision.transforms as T

# 默认均值与方差
RGB_MEAN = [87.01, 91.52, 83.51]
RGB_STD = [62.66, 54.58, 53.11]
UMBRA_SAR_MEAN = [36.42]
UMBRA_SAR_STD = [33.89]
CAPELLA_SAR_MEAN = [2353.35]
CAPELLA_SAR_STD = [4327.07]

class MaskGenerator:
    def __init__(self, input_size=512, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        assert input_size % mask_patch_size == 0
        assert mask_patch_size % model_patch_size == 0
        self.rand_size = input_size // mask_patch_size
        self.scale = mask_patch_size // model_patch_size
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        return mask  # shape: [H, W]

class M2Dataset(Dataset):
    def __init__(
        self,
        dataset_root="./data",
        img_size=256, # 512,32
        mask_patch_size=16, #32,#16
        model_patch_size=4,#4,#16
        mask_ratio=0.6,
        cache_file="file_list_cache.json",
    ):
        self.img_size = img_size
        self.dataset_root = dataset_root
        self.mask_generator = MaskGenerator(img_size, mask_patch_size, model_patch_size, mask_ratio)

        self.umbra_dir = os.path.join(self.dataset_root, "Umbra_patches")
        self.capella_dir = os.path.join(self.dataset_root, "Capella_patches")
        self.cache_file = os.path.join(self.dataset_root, cache_file)
        self.rgb_sar_pairs = []
        self._load_or_build_cache()

        # 4通道归一化
        self.transform = T.Compose([
            T.ToTensor(),  # (H,W,C) -> (C,H,W), range 0~255
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            # 归一化手动做
        ])

    def _load_or_build_cache(self):
        if os.path.exists(self.cache_file):
            print("Loading cached file list...")
            with open(self.cache_file, "r") as f:
                self.rgb_sar_pairs = json.load(f)
        else:
            print("Building file list from scratch...")
            entries = []
            for root_dir, sar_mean, sar_std, sar_dtype in [
                ("Umbra_patches", UMBRA_SAR_MEAN, UMBRA_SAR_STD, 'uint8'),
                ("Capella_patches", CAPELLA_SAR_MEAN, CAPELLA_SAR_STD, 'uint16')
            ]:
                rgb_dir = os.path.join(self.dataset_root, root_dir, "rgb")
                sar_dir = os.path.join(self.dataset_root, root_dir, "sar")
                for file in os.listdir(rgb_dir):
                    rgb_rel = os.path.relpath(os.path.join(rgb_dir, file), self.dataset_root)
                    sar_rel = os.path.relpath(os.path.join(sar_dir, file), self.dataset_root)
                    if os.path.exists(os.path.join(self.dataset_root, sar_rel)):
                        entries.append([rgb_rel, sar_rel, sar_mean, sar_std])
            with open(self.cache_file, "w") as f:
                json.dump(entries, f)
            self.rgb_sar_pairs = entries

    def __len__(self):
        return len(self.rgb_sar_pairs)

    def __getitem__(self, index):
        rgb_rel, sar_rel, sar_mean, sar_std = self.rgb_sar_pairs[index]
        rgb_path = os.path.join(self.dataset_root, rgb_rel)
        sar_path = os.path.join(self.dataset_root, sar_rel)

        rgb = tiff.imread(rgb_path).astype(np.float32)  # H,W,3
        sar = tiff.imread(sar_path).astype(np.float32)  # H,W
        image = np.concatenate([rgb, sar[..., None]], axis=-1)  # H,W,4
        image = self.transform(image)

        # 手动归一化（4通道）
        mean = torch.tensor(RGB_MEAN + sar_mean, dtype=torch.float32)[:, None, None]
        std = torch.tensor(RGB_STD + sar_std, dtype=torch.float32)[:, None, None]
        image = (image - mean) / std

        mask = self.mask_generator()
        return image, torch.tensor(mask, dtype=torch.long)  # (C,H,W), (H,W)

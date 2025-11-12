import pytorch_lightning as pl
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import random
import numpy as np
from typing import Optional

import tifffile as tiff
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

class DPGlobeDataModule(pl.LightningDataModule):
    def __init__(self, 
                 root_dir: str = '/mnt/data/MaRS/down_tasks/terratorch/data/dg_road',
                 crop_size: int = 256,
                 batch_size: int = 16,
                 num_workers: int = 4,
                 seed: int = 197):
        super().__init__()
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        
        # Define common transforms parameters
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        
        # Set seed for reproducibility
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
    
    def setup(self, stage: Optional[str] = None):
        # Define transforms
        self.train_transform = A.Compose([
            A.RandomCrop(width=self.crop_size, height=self.crop_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, p=0.5),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])
        
        self.val_transform = A.Compose([
            A.RandomCrop(width=self.crop_size, height=self.crop_size),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])

        self.test_transform = A.Compose([
            # A.RandomCrop(width=self.crop_size, height=self.crop_size),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])
        
        # Create datasets
        if stage == 'fit' or stage is None:
            self.train_dataset = DPGlobeDataset(
                root=self.root_dir + '/train',
                seed=self.seed,
                type='train',
                transform=self.train_transform
            )
            
            self.val_dataset = DPGlobeDataset(
                root=self.root_dir + '/test_label',
                seed=self.seed,
                type='val',
                transform=self.val_transform
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = DPGlobeDataset(
                root=self.root_dir + '/test_label',
                seed=self.seed,
                type='test',
                transform=self.test_transform
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


class DPGlobeDataset(torch.utils.data.Dataset):
    def __init__(self, root, seed=None, type=None, transform=None):
        imagelist = filter(lambda x: x.endswith('jpg'), os.listdir(root))
        imglist = list(map(lambda x: x[:-8], imagelist))
        self.ids = imglist
        self.root = root
        self.type = type
        self.transform = transform
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def __getitem__(self, index):
        id = self.ids[index]
        img = cv2.imread(os.path.join(self.root, f'{id}_sat.jpg'))
        mask = cv2.imread(os.path.join(self.root, f'{id}_mask.png'), cv2.IMREAD_GRAYSCALE)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            if self.type == 'train' or self.type == 'val':
                # print('pdb 1 ')
                # print(img.shape)
                # print(self.root)
                augmented = self.transform(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask'].long()
                # print('pdb 2')
                # print(img.shape)
            else:
                img = self.transform(image=img)['image']
                mask = torch.tensor(mask, dtype=torch.long)
        
        # Threshold mask
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0
        
        if self.type == 'train' or self.type == 'val':
            
            return img, mask
        else:
            return img, mask, f'{id}.png'
    
    def __len__(self):
        return len(self.ids)


class DFC23Track2Dataset(Dataset):
    def __init__(self, rgb_dir, sar_dir, label_dir, ids, transform=None):
        super().__init__()
        self.rgb_dir = rgb_dir
        self.sar_dir = sar_dir
        self.label_dir = label_dir
        self.ids = ids
        self.transform = transform
        self.label_max = 183.17412  # 用于归一化

    def __getitem__(self, index):
        id = self.ids[index]

        # 读取 RGB (H, W, 3)
        rgb_path = os.path.join(self.rgb_dir, f'{id}.tif')
        rgb = tiff.imread(rgb_path).astype(np.float32) / 255.0

        # 读取 SAR (H, W)
        sar_path = os.path.join(self.sar_dir, f'{id}.tif')
        sar = tiff.imread(sar_path).astype(np.float32)
        sar = np.expand_dims(sar, axis=2)

        # 读取 GT 高度图 (H, W)
        label_path = os.path.join(self.label_dir, f'{id}.tif')
        label = tiff.imread(label_path).astype(np.float32)
        label = np.where(label == 0, 0.0, label) / self.label_max  # 归一化到 0-1

        image = np.concatenate([rgb, sar], axis=2)

        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']

        return image, label

    def __len__(self):
        return len(self.ids)


class DFC23Track2DataModule(pl.LightningDataModule):
    def __init__(self, 
                 root_dir='/mnt/data/MaRS/down_tasks/terratorch/data/DFC23_track2/train',
                 test_dir=None,  # 新增：可选测试数据路径
                 crop_size=256,
                 batch_size=8,
                 num_workers=4,
                 seed=42):
        super().__init__()
        self.rgb_dir = os.path.join(root_dir, 'rgb')
        self.sar_dir = os.path.join(root_dir, 'sar')
        self.label_dir = os.path.join(root_dir, 'dsm')  # 高度标签也在 sar 中
        self.test_dir = test_dir  # 允许 test 用另一个路径

        self.crop_size = crop_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.rgb_mean = [0.3182, 0.3442, 0.2818]
        self.rgb_std = [0.1796, 0.1645, 0.1689]
        self.sar_mean = [0.2634]
        self.sar_std = [0.5957]

    def setup(self, stage=None):
        mean = self.rgb_mean + self.sar_mean
        std = self.rgb_std + self.sar_std

        if stage in ('fit', None):
            all_ids = [f[:-4] for f in os.listdir(self.rgb_dir) if f.endswith('.tif')]
            train_ids, val_ids = train_test_split(all_ids, test_size=0.2, random_state=self.seed)

            self.train_transform = A.Compose([
                A.RandomCrop(width=self.crop_size, height=self.crop_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])

            self.val_transform = A.Compose([
                A.CenterCrop(width=self.crop_size, height=self.crop_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])

            self.train_dataset = DFC23Track2Dataset(
                rgb_dir=self.rgb_dir,
                sar_dir=self.sar_dir,
                label_dir=self.label_dir,
                ids=train_ids,
                transform=self.train_transform
            )

            self.val_dataset = DFC23Track2Dataset(
                rgb_dir=self.rgb_dir,
                sar_dir=self.sar_dir,
                label_dir=self.label_dir,
                ids=val_ids,
                transform=self.val_transform
            )

        if stage in ('test', None):
            all_ids = [f[:-4] for f in os.listdir(self.rgb_dir) if f.endswith('.tif')]
            train_ids, test_ids = train_test_split(all_ids, test_size=0.2, random_state=self.seed)
            test_rgb_dir = os.path.join(self.test_dir, 'rgb') if self.test_dir else self.rgb_dir
            test_sar_dir = os.path.join(self.test_dir, 'sar') if self.test_dir else self.sar_dir
            test_label_dir = os.path.join(self.test_dir, 'dsm') if self.test_dir else self.label_dir
            # test_ids = [f[:-4] for f in os.listdir(test_rgb_dir) if f.endswith('.tif')]

            self.test_transform = A.Compose([
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])

            self.test_dataset = DFC23Track2Dataset(
                rgb_dir=test_rgb_dir,
                sar_dir=test_sar_dir,
                label_dir=test_label_dir,
                ids=test_ids,
                transform=self.test_transform
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)


class WHUCDDataset(Dataset):
    def __init__(self, root_dir, split_file, resize_size=256, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.resize_size = resize_size
        self.transform = transform

        with open(split_file, 'r') as f:
            self.ids = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        name = self.ids[index]
        imgA_path = os.path.join(self.root_dir, 'A', name)
        imgB_path = os.path.join(self.root_dir, 'B', name)
        label_path = os.path.join(self.root_dir, 'label', name)
        # print(f'Loading {name} from {imgA_path}, {imgB_path}, {label_path}')
        imgA = cv2.imread(imgA_path)
        
        imgB = cv2.imread(imgB_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
        imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)

        # Resize to resize_size × resize_size
        imgA = cv2.resize(imgA, (self.resize_size, self.resize_size), interpolation=cv2.INTER_LINEAR)
        imgB = cv2.resize(imgB, (self.resize_size, self.resize_size), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (self.resize_size, self.resize_size), interpolation=cv2.INTER_NEAREST)

        image = np.concatenate([imgA, imgB], axis=2)  # shape: H×W×6

        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask'].long()

        return image, label, name


class WHUCDDataModule(pl.LightningDataModule):
    def __init__(self, 
                 resize_size=256,
                 batch_size=8,
                 num_workers=4,
                 seed=42):
        super().__init__()
        self.root_dir = '/mnt/data/MaRS/down_tasks/terratorch/data/WHU-CD256-HANet'
        self.resize_size = resize_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.mean = [0.5] * 6
        self.std = [0.5] * 6

    def setup(self, stage: Optional[str] = None):
        train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])

        val_transform = A.Compose([
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])

        if stage == 'fit' or stage is None:
            self.train_dataset = WHUCDDataset(
                root_dir=os.path.join(self.root_dir, 'train'),
                split_file=os.path.join(self.root_dir, 'train', 'train.txt'),
                resize_size=self.resize_size,
                transform=train_transform
            )

            self.val_dataset = WHUCDDataset(
                root_dir=os.path.join(self.root_dir, 'val'),
                split_file=os.path.join(self.root_dir, 'val', 'val.txt'),
                resize_size=self.resize_size,
                transform=val_transform
            )

        if stage == 'test' or stage is None:
            self.test_dataset = WHUCDDataset(
                root_dir=os.path.join(self.root_dir, 'test'),
                split_file=os.path.join(self.root_dir, 'test', 'test.txt'),
                resize_size=self.resize_size,
                transform=val_transform
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)
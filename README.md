<div align="center">

<h1>MaRS: A Multi-Modality Very-High-Resolution Remote Sensing Foundation Model with Cross-Granularity Meta-Modality Learning</h1>

<h3>✨AAAI 2026✨</h3>

<div>
    <a href='https://github.com/WanderRainy/' target='_blank'>Ruoyu Yang</a><sup>1</sup>&emsp;
    <a>Yinhe Liu</a><sup>✉1</sup>&emsp;
    <a>Heng Yan</a><sup>1</sup>&emsp;
    <a>Yiheng Zhou</a><sup>1</sup>&emsp;
    <a>Yihan Fu</a><sup>1</sup>&emsp;
    <a>Han Luo</a><sup>1</sup>&emsp;
    <a href='https://rsidea.whu.edu.cn/' target='_blank'>Yanfei Zhong</a><sup>✉1</sup>&emsp;
</div>
<div>
    <sup>1</sup>Wuhan University&emsp;
</div>

<div>
    <h4 align="center">
        • <a href="https://rsidea.whu.edu.cn/mars.htm" target='_blank'>[Project]</a> • <a href="https://rsidea.whu.edu.cn/mars.pdf" target='_blank'>[paper]</a> • <a href="https://rsidea.whu.edu.cn" target='_blank'>[Research Group (RS-IDEA)]</a> •
    </h4>
</div>

<img src="https://github.com/user-attachments/assets/2be1af62-8b3d-439d-b23c-46e5af638569" width="100%"/>
Overall framework of MaRS and examples of downstream tasks.

</div>

---

## 📰 Latest News

- **Nov 2025** — MaRS paper accepted to **AAAI 2026**.  
- **Nov 2025** — Pretraining code and model weights officially released.

---

## 📦 Overview

**MaRS** is a large-scale multi-modality foundation model designed for very-high-resolution remote sensing imagery.  
It introduces **Cross-Granularity Meta-Modality Learning**, enabling robust representation learning across optical RGB and SAR modalities, at large spatial resolutions.

This repository provides:  
- Pretrained weights (`mars_base`, `mars_large`)  
- Pretraining pipeline (data processing, configuration, and scripts)  
- Instructions for loading MaRS using **timm** (compatible with SwinV2 architecture)

---

## 🔧 Using MaRS in Your Project

All pretrained weights are available at:  
<https://zenodo.org/records/17800805>

MaRS follows the **SwinV2** architecture and can be loaded directly using `timm==1.0.15`.

### ▶ Optical RGB Example

```python
backbone_mars = timm.create_model(
    'swinv2_base_window8_256',
    pretrained=False,
    features_only=True,
    in_chans=3,
    img_size=512,
    checkpoint_path='mars_base_rgb_encoder_only.pth'
)
```

### ▶ SAR Example

```python
backbone_mars = timm.create_model(
    'swinv2_base_window8_256',
    pretrained=False,
    features_only=True,
    in_chans=1,
    img_size=512,
    checkpoint_path='mars_base_sar_encoder_only.pth'
)
```

The pretrained backbone has been validated on a wide range of high-resolution optical and multi-modal downstream tasks (details in the paper).

---

## 🏗️ Pretraining Pipeline

This section describes how to reproduce MaRS pretraining.

---

### 1. Environment Setup

A minimal software environment used in our experiments:

```text
python   = 3.11.13
torch    = 2.7.0
tifffile = 2025.3.30
timm     = 1.0.15
```

---

### 2. Data Preparation

The full **MaRS-16M** pretraining corpus (~5 TB) is too large for public hosting.  
A **public experimental subset** will be released :<https://zenodo.org/records/17800805>.

<del>To request full dataset access for academic collaboration, please contact:</del>

```
<del>yangruoyu@whu.edu.cn</del>
```

Note: The dataset is currently under organization and is not publicly available. For collaboration inquiries, please feel free to contact us via email.


#### 2.1 Download & Organize Raw Data

```bash
mkdir -p ./data
# Place Umbra / Capella raw tiles into ./data
```

#### 2.2 Patch Extraction

Extract **1024 × 1024** training patches:

```bash
python ./data/split_patch.py
```

After extraction:

```text
data/
├── Capella_patches/
│   ├── rgb/
│   └── sar/
└── Umbra_patches/
    ├── rgb/
    └── sar/
```

- `rgb/`: optical patches  
- `sar/`: SAR patches  

---

### 3. Launching Pretraining

Example commands for 8×GPU single-node training using `torchrun`.

#### 3.1 MaRS-Base

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun \
    --nproc-per-node=8 \
    --nnodes=1 --node_rank=0 \
    --master_addr=localhost --master_port=12345 \
    main_pretrain.py \
    --model mars_base \
    --batch_size 16 \
    --num_workers 8 \
    --output_dir ./work_dirs/mars_base \
    --log_dir ./work_dirs/mars_base \
    --epochs 12 \
    --warmup_epochs 1
```

#### 3.2 MaRS-Large

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun \
    --nproc-per-node=8 \
    --nnodes=1 --node_rank=0 \
    --master_addr=localhost --master_port=12345 \
    main_pretrain.py \
    --model mars_large \
    --batch_size 12 \
    --num_workers 8 \
    --output_dir ./work_dirs/mars_large \
    --log_dir ./work_dirs/mars_large \
    --epochs 12 \
    --warmup_epochs 1
```

---

### 4. Converting MaRS Weights to Swin Format

To make MaRS weights directly loadable by SwinTransformer (and `timm`), convert them via:

```bash
python utils/convert_mars_checkpoints_to_swin.py
```

The released weights have already undergone this conversion.

---

## 📕 Downtasks Dataset
GUSO：Multi-modality Paired High-resolution Remote Sensing Dataset. [Under review]

EarthMiss: Missing Modality Land Cover Mapping. [Download](https://rsidea.whu.edu.cn/EarthMiss.html)

DFC25-T2: multimodal VHR dataset for all-weather disaster response. [Download](https://github.com/ChenHongruixuan/BRIGHT)

SARDet-100k: SAR Modality Object Detection Dataset. [Download](https://github.com/zcablii/SARDet_100K)

UBC-V2: Multi-modality High-resolution Remote Sensing Building Detection Dataset. [Download](https://github.com/AICyberTeam/UBC-dataset/tree/UBCv2)

UBC: Multi-modality High-resolution Remote Sensing Building Height Estimation Dataset. [Download](https://github.com/AICyberTeam/UBC-dataset)

WHU-CD: High-resolution Remote Sensing Change Detection Dataset. [Download](https://gpcv.whu.edu.cn/data/building_dataset.html)

DeepGlobe: High-resolution Remote Sensing Road Extraction Dataset. [Download](https://www.eotdl.com/datasets/DeepGlobeRoadExtraction)

---

## 📖 Citation

If you find **MaRS** useful in your research, please cite:

```bibtex
@inproceedings{yang2026mars,
  title={MaRS: A Multi-Modality Very-High-Resolution Remote Sensing Foundation Model with Cross-Granularity Meta-Modality Learning},
  author={Ruoyu Yang and Yinhe Liu and Heng Yan and Yiheng Zhou and Yihan Fu and Han Luo and Yanfei Zhong},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

---

## © Copyright & Usage

This method is copyrighted by the **Intelligent Remote Sensing Data Extraction, Analysis and Application Research Group (RSIDEA)**  
<http://rsidea.whu.edu.cn/>  
affiliated with the **State Key Laboratory of Information Engineering in Surveying, Mapping and Remote Sensing (LIESMARS), Wuhan University**.

**MaRS is released strictly for academic research purposes.**

---

# MaRS  
*A multi-modality very-high-resolution remote sensing foundation model with Cross-Granularity Meta-Modality Learning*  

[![Project Status](https://img.shields.io/badge/status-active-development-brightgreen)](https://github.com/WanderRainy/MaRS)  
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  
[![GitHub stars](https://img.shields.io/github/stars/WanderRainy/MaRS.svg?style=social&label=Star)](https://github.com/WanderRainy/MaRS)  

---

## 🚀 Quick Links

- **Paper PDF**: [Download here](MaRS_AAAI26.pdf)  
- **Code Repository**: [GitHub](https://github.com/WanderRainy/MaRS)  
- **Dataset Page**: [MaRS-16M Dataset](https://rsidea.whu.edu.cn/mars.htm)  
- **Project Homepage**: [RS-IDEA MaRS Project](https://rsidea.whu.edu.cn)  

---

## 📌 Project Overview  
MaRS is developed by the RS‑IDEA Lab at Wuhan University.  
It aims to build a powerful foundation model for very-high-resolution (VHR) remote sensing data by leveraging multi-modality (SAR + optical) and introducing advanced learning strategies:

- A large-scale paired VHR SAR–Optical dataset **MaRS-16M** (≈ 16.8 million patch pairs).  
- **Cross-Granularity Contrastive Learning (CGCL)** to align patch- and image-level semantics across modalities.  
- **Meta-Modality Attention (MMA)** to unify heterogeneous modality representations via alternating intra-/cross-modality attention.  
- Extensive evaluation across **nine** VHR multi-modality downstream tasks, demonstrating strong transfer ability of the MaRS model.

---

## 🎯 Key Features  
- ✅ Supports both SAR & Optical modalities at very high resolution.  
- ✅ Robust to cross-modality alignment issues (geometric distortion, missing modality).  
- ✅ Acts as a general pretrained backbone for classification, detection, segmentation, change detection, height estimation, mapping, and other tasks.  
- ✅ Open-source code + dataset (with licensing info) for reproducibility.

---

## 📚 Dataset (MaRS-16M)  
| Metric        | Value                        |
|---------------|------------------------------|
| Number of pairs | 16,785,168 SAR–Optical patches |
| Resolution       | ~0.35 m GSD                   |
| SAR sensors      | Umbra, Capella (X-band HH/VV) |
| Patch size       | 512 × 512                      |
| Coverage         | Global land cover, urban, disaster |
| Use case         | Self-supervised pre-training on VHR multi-modality data |

---

## 🧠 Model & Method  
### Architecture  
MaRS uses dual encoders (SwinV2 for optical, SwinV2 for SAR) → Meta-Modality Attention (MMA) Transformer → light task-specific heads.  
### Pre-training Strategy  
1. CGCL: patch-to-patch, patch-to-image, image-to-image contrastive training.  
2. Masked image modelling per modality branch.  
3. Continued pre-training on large VHR optical corpora for further refinement.  
Inputs: 512×512 patches; Masking ratio ≈ 60%; Hardware: 8×A800 GPUs (example)  
### Downstream Tasks  
Includes registration, modality translation, missing-modality mapping, target detection, building detection, height estimation, change detection, road extraction, damage assessment.

---

## 📊 Results Summary  
A selection of results:  
- Cross-Modality Registration (GUSO): RMSE ≈ 2.83  
- Modality-Missing Mapping (EarthMiss): mIoU ≈ 49.90  
- Cross-Modality Translation (GUSO): PSNR ≈ 20.69  
- SAR Target Detection (ARTDet / SARDet-100K): mAP ≈ 55.40  
- … and others as detailed in the paper.

---

## 📦 Installation & Usage  
```bash
# Clone repository
git clone https://github.com/WanderRainy/MaRS.git
cd MaRS

# Install requirements (example)
pip install -r requirements.txt

# Example usage
python train_pretrain.py --config configs/mars_pretrain.yaml
python downstream_task.py --task building_detection --pretrained model/mars.pth

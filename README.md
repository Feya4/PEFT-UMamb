# PEFT-UMamba: Parameter-Efficient Fine-Tuning for Mamba-Based Medical Image Segmentation
## Overview

PEFT-UMamba is a parameter-efficient fine-tuning framework for medical image segmentation...

## Installation

```bash
pip install torch==2.0.1 torchvision==0.15.2
pip install causal-conv1d==1.1.1
pip install mamba-ssm
pip install timm nibabel monai scipy scikit-image
pip install opencv-python matplotlib tqdm
```

## Pretrained Backbone

Download VMamba-Tiny pretrained on ImageNet-1K:
mkdir -p pretrained
wget https://github.com/MzeroMiko/VMamba/releases/download/v2/vssm_tiny_0230_ckpt_epoch_262.pth \
     -O pretrained/vmamba_tiny.pth
```

## Dataset Preparation

### AMOS22 AbdomenMRI
Download from [AMOS22 Challenge](https://amos22.grand-challenge.org/) and organize:
```
data/Dataset702_AbdomenMR/
├── imagesTr/
├── labelsTr/
├── imagesVal/
└── labelsVal/
```
### Kvasir-SEG
Download from [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) and organize:
```
data/kvasir/
├── images/
└── masks/
```
### MICCAI EndoVis17
Download from [EndoVis17 Challenge](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/) and organize:
```
data/Dataset704_Endovis17_nii/
├── imagesTr/
├── labelsTr/
├── imagesVal/
└── labelsVal/
```

### NeurIPS CellSeg
Download from [NeurIPS CellSeg Challenge](https://neurips22-cellseg.grand-challenge.org/) and organize:
```
data/Dataset703_NeurIPSCell/
├── imagesTr/
├── labelsTr/
├── imagesVal/
├── labelsVal/
└── labelsVal-instance-mask/
```

---

## Training

### AMOS22 AbdomenMRI (16-class)

```bash
CUDA_VISIBLE_DEVICES=0 python train_noSC.py \
  --dataset dataset702 \
  --data_root data/Dataset702_AbdomenMR \
  --output_dir outputs \
  --exp_name peft_umamba_amos22 \
  --epochs 150
```

### Kvasir-SEG (binary polyp)

```bash
CUDA_VISIBLE_DEVICES=0 python train_noSC.py \
  --dataset kvasir \
  --data_root data/kvasir \
  --output_dir outputs \
  --exp_name peft_umamba_kvasir \
  --epochs 200
```

### MICCAI EndoVis17 (surgical instruments)

```bash
CUDA_VISIBLE_DEVICES=0 python train_noSC.py \
  --dataset dataset704 \
  --data_root data/Dataset704_Endovis17_nii \
  --output_dir outputs \
  --exp_name peft_umamba_endovis17 \
  --epochs 150
```

### NeurIPS CellSeg (microscopy)

```bash
CUDA_VISIBLE_DEVICES=0 python train_noSC.py \
  --dataset dataset703 \
  --data_root data/Dataset703_NeurIPSCell \
  --output_dir outputs \
  --exp_name peft_umamba_neurips \
  --epochs 200
```

## Evaluation

### Semantic DSC evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python eval_checkpoint_kvasir.py \
  --checkpoint outputs/peft_umamba_amos22/best_model.pth \
  --dataset dataset702 \
  --data_root data/Dataset702_AbdomenMR
```
### Instance F1 evaluation (NeurIPS CellSeg)

```bash
CUDA_VISIBLE_DEVICES=0 python eval_instance_f1_v2.py \
  --checkpoint outputs/peft_umamba_neurips/best_model.pth \
  --data_root data/Dataset703_NeurIPSCell
```
## Pretrained Models

| Dataset | DSC | Download |
|---------|-----|----------|
| AMOS22 AbdomenMRI | 0.6916 | [Google Drive](https://drive.google.com/XXXX) |
| Kvasir-SEG | 0.8034 | [Google Drive](https://drive.google.com/XXXX) |
| EndoVis17 | 0.7781 | [Google Drive](https://drive.google.com/XXXX) |
| NeurIPS CellSeg | 0.8892/0.6347 | [Google Drive](https://drive.google.com/XXXX) |

## Project Structure

```
PEFT-UMamba/
├── train_noSC.py          # Main training script
├── eval_checkpoint_kvasir.py  # Evaluation script
├── eval_instance_f1_v2.py     # Instance F1 evaluation
├── visualizaion_zoom.py       # Visualization script
├── configs/
│   └── config.py          # Dataset configurations
├── models/
│   └── model.py           # PEFT-UMamba model
├── data/
│   └── dataset.py         # Dataset loaders
├── utils/
│   └── utils.py           # Training utilities
└── pretrained/
    └── vmamba_tiny.pth    # VMamba-Tiny backbone
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{peft_umamba2025,
  title   = {PEFT-UMamba: Parameter-Efficient Fine-Tuning for 
             Mamba-Based Medical Image Segmentation via 
             Supplementary Scan},
  author  = {Feidu Akmel and Xun Gong},
  %journal = {},
  year    = {2025}
}
```

## Acknowledgements

This work builds upon [Swin-UMamba](https://github.com/JiarunLiu/Swin-UMamba). We sincerely thank the authors for their wonderful contributions.

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

# PEFT-UMamba: Parameter-Efficient Mamba Adaptation in Medical Image Segmentation

## Overview

PEFT-UMamba adapts an ImageNet-pretrained VMamba-Tiny encoder to medical image
segmentation while keeping **every pretrained weight frozen**. Its core is
**Supplementary Scan**: the hidden state of each SS2D block is expanded from
K = 16 to K + K′ dimensions, and only the K′ = 4 supplementary dimensions are
trained (0.17M parameters). Supplementary dimensions are initialised with
Neighbourhood Initialisation, and their output projection starts at zero, so the
network is identical to the pretrained backbone before the first update.

The framework also includes a Sequential Context Aggregation Module (SCAM,
bidirectional GRU over adjacent slices), Skip Attention Gates, a VSS-block
decoder and deep supervision at three decoder resolutions.

## Installation

```bash
conda create -n peft_umamba python=3.10 -y
conda activate peft_umamba
pip install torch==2.0.1 torchvision==0.15.2
pip install causal-conv1d==1.1.1
pip install mamba-ssm
pip install numpy scipy nibabel pillow
```

The `mamba-ssm` CUDA kernel is required for practical training speed;
`check_setup.py` reports whether it is active.

## Pretrained Backbone

PEFT-UMamba uses the **original VMamba-Tiny** release (K = 16, depths 2-2-9-2),
the same backbone as Swin-UMamba:

```bash
mkdir -p data/vmamba
wget https://github.com/MzeroMiko/VMamba/releases/download/%2320240218/vssmtiny_dp01_ckpt_epoch_292.pth \
     -O data/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth
```

Later VMamba checkpoints (e.g. `vssm_tiny_0230_ckpt_epoch_262.pth`,
`vssm1_tiny_0230s_ckpt_epoch_264.pth`) use a different block design and are
rejected by the loader. Verify the setup before training:

```bash
python check_setup.py
```

Every line must report `[OK]`: all 208 encoder tensors load, only the
supplementary parameters are trainable, and at initialisation the adapted
encoder reproduces the frozen backbone exactly.

## Dataset Preparation

Volumetric and nnU-Net-style datasets use the nnU-Net raw layout
(`dataset.json`, `imagesTr/<case>_0000<ext>`, `labelsTr/<case><ext>`).
Only `imagesTr`/`labelsTr` are read. A fixed 85/15 train/validation split is
drawn once per dataset and saved to `<data_root>/splits_peft_umamba.json`;
every method and every seed uses the same split.

### AMOS22 AbdomenMRI (Dataset702)
From the [AMOS22 Challenge](https://amos22.grand-challenge.org/):
```
data/Dataset702_AbdomenMR/
├── dataset.json
├── imagesTr/
└── labelsTr/
```
Volumes are resampled to 1.5 mm isotropic, normalised and cached under
`_cache_peft_umamba/` on first use.

### Kvasir-SEG
From [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/):
```
data/kvasir/
├── images/
└── masks/
```
(`data/kvasir/Kvasir-SEG/images` and `.../masks` also work.)

### MICCAI EndoVis17 (Dataset704)
From the [EndoVis17 Challenge](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/):
```
data/Dataset704_Endovis17/
├── dataset.json
├── imagesTr/
└── labelsTr/
```
Following Swin-UMamba, frames are split at random (not by video sequence).

### NeurIPS CellSeg (Dataset703)
From the [NeurIPS CellSeg Challenge](https://neurips22-cellseg.grand-challenge.org/):
```
data/Dataset703_NeurIPSCell/
├── dataset.json
├── imagesTr/
└── labelsTr/      # 3 classes: background / cell interior / cell boundary
```
Instance F1 (IoU > 0.5) is computed from connected components of the
cell-interior class in prediction and ground truth.

## Training and Evaluation

One script trains, evaluates the **final-epoch** model on the validation split
and records the results. All runs follow the same protocol: AdamW (lr 2e-4,
weight decay 0.05), 5-epoch warm-up + polynomial decay (power 0.9), gradient
clipping 1.0, CE + Dice loss with deep supervision (0.2/0.3/0.5), 200 epochs on
Kvasir-SEG and 150 otherwise, batch 16 ordered adjacent slices for volumes and
8 for 2-D images.

```bash
# AMOS22 AbdomenMRI
python train.py --dataset dataset702 --data_root data/Dataset702_AbdomenMR --seed 0

# Kvasir-SEG
python train.py --dataset kvasir     --data_root data/kvasir                --seed 0

# MICCAI EndoVis17
python train.py --dataset dataset704 --data_root data/Dataset704_Endovis17  --seed 0

# NeurIPS CellSeg
python train.py --dataset dataset703 --data_root data/Dataset703_NeurIPSCell --seed 0
```

Comparison methods and ablations use the same script:

```bash
# adaptation strategies on the same frozen encoder (Table 6)
python train.py --dataset kvasir --data_root data/kvasir --method frozen|lora|full_ft|mona|lorand|e3va

# component ablations (Table 5) and K′ sensitivity (Table 7)
python train.py ... --no_supp_scan | --no_scam | --no_skip_gate | --no_deep_sup | --supp_init s4d
python train.py ... --k_prime 2|8|16
```

`run_all.sh` runs every experiment with four seeds (edit the data paths first).

### Results

Each run writes `outputs/<experiment>/train.log`, `config.json`,
`results.json` (Dice ± std across cases, per-class Dice, HD95 for volumes,
instance F1 for CellSeg, parameter counts, training memory, inference time)
and `final.pth`, and appends one line to `outputs/summary.csv`.

```bash
python summarize.py          # mean ± std over seeds for every experiment
```

## Pretrained Models

Trained checkpoints will be released with the published version of the paper.

## Project Structure

```
PEFT-UMamba/
├── train.py              # training + final evaluation for every table
├── check_setup.py        # verifies backbone loading and initial equivalence
├── peft_baselines.py     # Mona, LoRand and E3VA baselines
├── summarize.py          # aggregates results over seeds
├── run_all.sh            # all experiments of the paper
├── configs/
│   └── config.py         # training protocol and dataset settings
├── models/
│   └── model.py          # PEFT-UMamba model and verified checkpoint loader
├── data/
│   ├── dataset.py        # dataset loading, ordered slice runs, inference
│   └── vmamba/
│       └── vssmtiny_dp01_ckpt_epoch_292.pth
└── utils/
    └── utils.py          # loss, optimiser, schedule, metrics
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{akmel2026peftumamba,
  title   = {{PEFT-UMamba}: Parameter-Efficient Mamba Adaptation in Medical Image Segmentation},
  author  = {Akmel, Feidu and Gong, Xun and Meng, Fanman and Gedamu, Kumie and Wake, Negalign and Bentahar, Jamal},
  journal = {Knowledge-Based Systems},
  note    = {Under review},
  year    = {2026}
}
```

## Acknowledgements

This work builds on [VMamba](https://github.com/MzeroMiko/VMamba),
[Swin-UMamba](https://github.com/JiarunLiu/Swin-UMamba) and the state-expansion
PEFT and Neighbourhood Initialisation of
MambaPEFT (Yoshimura et al., ICLR 2025).
We thank the authors for releasing their code and models.

## License

This project is released under the MIT License; see [LICENSE](LICENSE).

"""
configs/config.py — one training protocol for all datasets (manuscript §4.2).

  Optimiser      AdamW, lr 2e-4, weight decay 0.05, gradient clipping 1.0
  Schedule       linear warm-up (5 epochs) then polynomial decay (power 0.9)
  Loss           CE + Dice, deep supervision at three decoder resolutions,
                 weights 0.2 / 0.3 / 0.5 (deepest → shallowest)
  Epochs         150 for every dataset; for volumetric
                 datasets an epoch is a fixed 250 iterations (nnU-Net
                 convention), each drawing a random run of adjacent slices
  Batch          16 ordered adjacent slices (volumetric MRI/CT), 8 otherwise
  Augmentation   h/v flips, rotation up to 90°, colour jitter (RGB only)
  Seeds          4 independent runs; the FINAL-epoch model is reported
                 (no model selection on the validation set)

Everything that changes between Table rows (method, K′, ablation switches)
is a command-line flag of train.py — the protocol itself never changes.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# =========================================================================== #
#  Sub-configs
# =========================================================================== #

@dataclass
class SSMConfig:
    d_state: int = 16                  # K  (VMamba-Tiny)


@dataclass
class PEFTConfig:
    use_supplementary_scan: bool = True
    supp_state_dim: int = 4            # K′
    supp_init: str = "neighbourhood"   # "neighbourhood" | "s4d"  (Table 5, F)
    no_skip_gate: bool = False         # Table 5, D
    use_lora: bool = False             # Table 6
    lora_rank: int = 4
    lora_alpha: float = 8.0


@dataclass
class ModelConfig:
    num_classes: int = 16
    img_size: int = 224
    depths: List[int] = field(default_factory=lambda: [2, 2, 9, 2])
    feat_dims: List[int] = field(default_factory=lambda: [96, 192, 384, 768])
    pretrained_path: Optional[str] = None
    freeze_encoder: bool = True
    decoder_depth: int = 2
    use_scam: bool = True              # Table 5, C
    scam_unordered: bool = False       # control: SCAM sees length-1 sequences
    scam_hidden: int = 64
    ssm: SSMConfig = field(default_factory=SSMConfig)
    peft: PEFTConfig = field(default_factory=PEFTConfig)


@dataclass
class DataConfig:
    dataset: str = "dataset702"
    data_root: str = "./data"
    kind: str = "volume"               # "volume" (NIfTI 3-D) | "image2d"
    modality: str = "mri"              # mri | ct | rgb
    img_size: int = 224
    num_classes: int = 16
    ordered: bool = True               # batches are runs of adjacent slices
    spacing: float = 1.5               # resampling (mm), volumes only
    resample_z: bool = True            # False: in-plane only (thick-slice MRI)
    val_fraction: float = 0.15
    split_seed: int = 0                # fixed split shared by every method/seed
    num_workers: int = 8
    instance_f1: bool = False          # NeurIPS CellSeg


@dataclass
class TrainConfig:
    epochs: int = 150
    batch_size: int = 16
    base_lr: float = 2e-4
    min_lr: float = 0.0
    weight_decay: float = 0.05
    gradient_clip: float = 1.0
    warmup_epochs: int = 5
    poly_power: float = 0.9
    ce_weight: float = 1.0
    dice_weight: float = 1.0
    deep_sup: bool = True              # Table 5, E
    deep_sup_weights: Tuple[float, float, float] = (0.2, 0.3, 0.5)
    rotation_deg: float = 90.0
    use_amp: bool = True
    eval_every: int = 10               # monitoring only; final epoch is reported
    seed: int = 42
    output_dir: str = "./outputs"
    experiment_name: str = "peft_umamba"
    iters_per_epoch: Optional[int] = None   # None = one pass over the data
    max_train_steps: Optional[int] = None   # smoke tests only


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# =========================================================================== #
#  Datasets
# =========================================================================== #

DATASETS = {
    # key          kind       modality img  ordered epochs batch instF1
    "dataset701": ("volume",  "ct",   224, True,  150, 16, False),
    "dataset702": ("volume",  "mri",  224, True,  150, 16, False),
    "dataset703": ("image2d", "rgb",  352, False, 150,  8, True),
    "dataset704": ("image2d", "rgb",  224, False, 150,  8, False),
    "kvasir":     ("image2d", "rgb",  352, False, 150,  8, False),
    "dataset705": ("volume",  "mri",  224, True,  150, 16, False),   # ACDC
}

# volumetric datasets: fixed iterations per epoch (random slice runs)
VOLUME_ITERS_PER_EPOCH = 250

# thick-slice MRI: resample in-plane only, keep native slice spacing
NO_Z_RESAMPLE = {"dataset705"}

ALIASES = {
    "abdomenct": "dataset701", "synapse": "dataset701",
    "abdomenmr": "dataset702", "amos": "dataset702", "amos22": "dataset702",
    "neurips": "dataset703", "cellseg": "dataset703", "cell": "dataset703",
    "endovis": "dataset704", "endovis17": "dataset704",
    "kvasir_seg": "kvasir", "kvasirseg": "kvasir", "polyp": "kvasir",
    "acdc": "dataset705", "cardiac": "dataset705",
}

# Number of classes INCLUDING background; overridden by dataset.json if present.
DEFAULT_NUM_CLASSES = {"dataset701": 14, "dataset702": 16, "dataset703": 3,
                       "dataset704": 8, "kvasir": 2, "dataset705": 4}


def resolve_dataset_key(name: str) -> str:
    key = name.lower().replace("-", "").replace(" ", "")
    key = ALIASES.get(key, key)
    if key not in DATASETS:
        raise ValueError(f"unknown dataset '{name}'. Choose from "
                         f"{sorted(DATASETS)} or aliases {sorted(ALIASES)}")
    return key


def config_from_dataset_name(name: str, data_root: str) -> Config:
    key = resolve_dataset_key(name)
    kind, modality, img, ordered, epochs, batch, inst = DATASETS[key]
    cfg = Config()
    cfg.data.dataset = key
    cfg.data.data_root = data_root
    cfg.data.kind = kind
    cfg.data.modality = modality
    cfg.data.img_size = img
    cfg.data.ordered = ordered
    cfg.data.instance_f1 = inst
    cfg.data.resample_z = key not in NO_Z_RESAMPLE
    cfg.data.num_classes = DEFAULT_NUM_CLASSES[key]
    cfg.model.num_classes = cfg.data.num_classes
    cfg.model.img_size = img
    cfg.train.epochs = epochs
    cfg.train.batch_size = batch
    cfg.train.iters_per_epoch = VOLUME_ITERS_PER_EPOCH if kind == "volume" else None
    cfg.train.experiment_name = key
    return cfg

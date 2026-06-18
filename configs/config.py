""" 

from dataclasses import dataclass, field
from typing import List, Optional


# =========================================================================== #
#  Sub-configs
# =========================================================================== #

@dataclass
class SSMConfig:
    d_state:       int   = 16       # base SSM state dimension K
    d_state_supp:  int   = 4        # supplementary dimension K
    d_conv:        int   = 4
    expand:        int   = 2
    dt_min:        float = 0.001
    dt_max:        float = 0.1
    dt_init_floor: float = 1e-4


@dataclass
class PEFTConfig:
    use_supplementary_scan: bool  = True
    supp_state_dim:         int   = 4       # K
    ni_noise_std:           float = 1e-6

    use_lora:     bool  = False
    lora_rank:    int   = 4
    lora_alpha:   float = 8.0
    lora_dropout: float = 0.0

    train_supplementary: bool = True
    train_head:          bool = True
    train_decoder_norm:  bool = True
    train_patch_expand:  bool = True


@dataclass
class ModelConfig:
    backbone:    str       = "vmamba_tiny"
    img_size:    int       = 224
    in_channels: int       = 3
    num_classes: int       = 9

    depths:    List[int] = field(default_factory=lambda: [2, 2, 9, 2])
    feat_dims: List[int] = field(default_factory=lambda: [96, 192, 384, 768])

    pretrained_path: Optional[str] = None
    freeze_encoder:  bool          = True

    ssm:  SSMConfig  = field(default_factory=SSMConfig)
    peft: PEFTConfig = field(default_factory=PEFTConfig)


@dataclass
class DataConfig:
    dataset:         str   = "dataset702"
    data_root:       str   = "./data"
    img_size:        int   = 224
    num_classes:     int   = 16
    modality:        str   = "mri"
    train_val_split: float = 0.85
    num_workers:     int   = 4


@dataclass
class TrainConfig:
    stage:         str   = "finetune"
    epochs:        int   = 150
    warmup_epochs: int   = 5

    optimizer:     str   = "adamw"
    base_lr:       float = 1e-4
    min_lr:        float = 1e-6
    weight_decay:  float = 1e-4
    gradient_clip: float = 1.0

    supp_lr_scale:    float = 3.0    # supplementary SSM params: lr  3
    decoder_lr_scale: float = 10.0   # decoder (random init): lr  10
    head_lr_scale:    float = 10.0

    scheduler:    str   = "poly"
    poly_power:   float = 0.9

    loss:         str   = "ce_dice"
    dice_weight:  float = 0.5
    ce_weight:    float = 0.5
    focal_gamma:  float = 2.0
    bg_weight:    float = 0.1

    batch_size:         int = 24
    accumulation_steps: int = 1

    log_interval:  int = 50
    eval_interval: int = 1
    save_interval: int = 5
    output_dir:    str = "./outputs"
    experiment_name: str = "peft_umamba"

    use_amp:     bool = True
    distributed: bool = False
    seed:        int  = 42


@dataclass
class MIMConfig:
    mask_ratio:    float = 0.60
    patch_size:    int   = 4
    loss:          str   = "l1"
    epochs:        int   = 50
    warmup_epochs: int   = 5
    base_lr:       float = 1e-4
    min_lr:        float = 1e-6
    weight_decay:  float = 0.05
    batch_size:    int   = 32
    peft_only:     bool  = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data:  DataConfig  = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    mim:   MIMConfig   = field(default_factory=MIMConfig)

    def __post_init__(self):
        self.model.num_classes = self.data.num_classes
        self.model.img_size    = self.data.img_size


# =========================================================================== #
#  Pre-built dataset configs
# =========================================================================== #

def get_dataset701_config(data_root: str = "./data/Dataset701_AbdomenCT") -> Config:
    cfg = Config()
    cfg.data.dataset      = "dataset701"
    cfg.data.num_classes  = 9
    cfg.data.img_size     = 224
    cfg.data.modality     = "ct"
    cfg.data.data_root    = data_root
    cfg.model.num_classes = 9
    cfg.model.img_size    = 224
    cfg.train.epochs      = 150
    cfg.train.base_lr     = 1e-4
    cfg.train.batch_size  = 16
    cfg.train.loss        = "ce_dice"
    cfg.train.experiment_name = "peft_umamba_dataset701_ct"
    return cfg


def get_dataset702_config(data_root: str = "/workdir1.8t/fei27/CGT/peft_umamba/peft_umamba_2/data/Dataset702_AbdomenMR") -> Config:
    cfg = Config()
    cfg.data.dataset       = "dataset702"
    cfg.data.num_classes   = 16
    cfg.data.img_size      = 224
    cfg.data.modality      = "mri"
    cfg.data.data_root     = data_root
    cfg.model.num_classes  = 16
    cfg.model.img_size     = 224
    cfg.train.epochs       = 150
    # Optimizer: AdamW base; decoder gets higher LR via decoder_lr_scale
    cfg.train.optimizer    = "adamw"
    cfg.train.base_lr      = 1e-4
    cfg.train.weight_decay = 1e-2           #  was 1e-4; stronger regularisation stops overfitting
    cfg.train.supp_lr_scale    = 3.0
    cfg.train.decoder_lr_scale = 10.0
    cfg.train.scheduler    = "poly"
    cfg.train.poly_power   = 0.9
    cfg.train.batch_size   = 24
    cfg.train.loss         = "ce_dice"
    cfg.train.dice_weight  = 0.5
    cfg.train.ce_weight    = 0.5
    cfg.train.bg_weight    = 0.1
    cfg.train.experiment_name = "peft_umamba_dataset702_mr"
    return cfg


def get_dataset704_config(data_root: str = "./data/Dataset704_Endovis17") -> Config:
    cfg = Config()
    cfg.data.dataset      = "dataset704"
    cfg.data.num_classes  = 8
    cfg.data.img_size     = 352
    cfg.data.modality     = "endoscopy"
    cfg.data.data_root    = data_root
    cfg.model.num_classes = 8
    cfg.model.img_size    = 352
    cfg.train.epochs      = 100
    cfg.train.base_lr     = 3e-4
    cfg.train.batch_size  = 8
    cfg.train.loss        = "focal_dice"
    cfg.train.experiment_name = "peft_umamba_dataset704_endovis"
    return cfg


def get_kvasir_config(data_root: str = "./data/kvasir") -> Config:
    cfg = Config()
    cfg.data.dataset      = "kvasir"
    cfg.data.num_classes  = 2
    cfg.data.img_size     = 352
    cfg.data.modality     = "endoscopy"
    cfg.data.data_root    = data_root
    cfg.model.num_classes = 2
    cfg.model.img_size    = 352
    cfg.train.epochs      = 100
    cfg.train.base_lr     = 3e-4
    cfg.train.batch_size  = 8
    cfg.train.loss        = "focal_dice"
    cfg.train.experiment_name = "peft_umamba_kvasir"
    return cfg


# Aliases
get_synapse_config   = get_dataset701_config
get_amos_config      = get_dataset702_config
get_endovis_config   = get_dataset704_config
get_clinicdb_config  = get_kvasir_config
get_endoscopy_config = get_kvasir_config


def config_from_dataset_name(name: str, data_root: str = None) -> Config:
    from data.dataset import resolve_dataset_key
    key = resolve_dataset_key(name)
    builders = {
        "dataset701": get_dataset701_config,
        "dataset702": get_dataset702_config,
        "dataset704": get_dataset704_config,
        "kvasir":     get_kvasir_config,
    }
    return builders[key](data_root) if data_root else builders[key]() """
"""
configs/config.py - Hyperparameter configuration for PEFT-UMamba.

Key changes from previous version

   gradient_clip  5.0  1.0   (prevent large early-training gradients)
   warmup_epochs    10  5     (faster PEFT ramp-up)
   eval_interval     1  5     (volumetric eval is expensive; skip early epochs)
   save_interval    10  5     (catch best model more frequently)
   Separate LR groups in build_optimizer (see utils.py):
      supp params  : lr  3  (need to learn fast from frozen backbone)
      decoder      : lr  1
      head         : lr  1
"""
""" 
from dataclasses import dataclass, field
from typing import List, Optional


# =========================================================================== #
#  Sub-configs
# =========================================================================== #

@dataclass
class SSMConfig:
    d_state:       int   = 16       # base SSM state dimension K
    d_state_supp:  int   = 4        # supplementary dimension K
    d_conv:        int   = 4
    expand:        int   = 2
    dt_min:        float = 0.001
    dt_max:        float = 0.1
    dt_init_floor: float = 1e-4


@dataclass
class PEFTConfig:
    use_supplementary_scan: bool  = True
    supp_state_dim:         int   = 4       # K
    ni_noise_std:           float = 1e-6

    use_lora:     bool  = False
    lora_rank:    int   = 4
    lora_alpha:   float = 8.0
    lora_dropout: float = 0.0

    supp_init:    str   = "neighbourhood"  # NI ablation: neighbourhood|zero|random_normal|xavier|copy_frozen
    use_sdlora:   bool  = False            # SDLoRA ablation

    train_supplementary: bool = True
    train_head:          bool = True
    train_decoder_norm:  bool = True
    train_patch_expand:  bool = True


@dataclass
class ModelConfig:
    backbone:    str       = "vmamba_tiny"
    img_size:    int       = 224
    in_channels: int       = 3
    num_classes: int       = 9

    depths:    List[int] = field(default_factory=lambda: [2, 2, 9, 2])
    feat_dims: List[int] = field(default_factory=lambda: [96, 192, 384, 768])

    pretrained_path: Optional[str] = None
    freeze_encoder:  bool          = True

    ssm:  SSMConfig  = field(default_factory=SSMConfig)
    peft: PEFTConfig = field(default_factory=PEFTConfig)


@dataclass
class DataConfig:
    dataset:         str   = "dataset702"
    data_root:       str   = "./data"
    img_size:        int   = 224
    num_classes:     int   = 16
    modality:        str   = "mri"
    train_val_split: float = 0.85
    num_workers:     int   = 4


@dataclass
class TrainConfig:
    stage:         str   = "finetune"
    epochs:        int   = 150
    warmup_epochs: int   = 5

    optimizer:     str   = "adamw"
    base_lr:       float = 1e-4
    min_lr:        float = 1e-6
    weight_decay:  float = 1e-4
    gradient_clip: float = 1.0

    supp_lr_scale:    float = 3.0    # supplementary SSM params: lr  3
    decoder_lr_scale: float = 10.0   # decoder (random init): lr  10
    head_lr_scale:    float = 10.0

    scheduler:    str   = "poly"
    poly_power:   float = 0.9

    loss:         str   = "ce_dice"
    dice_weight:  float = 0.5
    ce_weight:    float = 0.5
    focal_gamma:  float = 2.0
    bg_weight:    float = 0.1

    batch_size:         int = 24
    accumulation_steps: int = 1

    log_interval:  int = 50
    eval_interval: int = 1
    save_interval: int = 5
    output_dir:    str = "./outputs"
    experiment_name: str = "peft_umamba"

    use_amp:     bool = True
    distributed: bool = False
    seed:        int  = 42


@dataclass
class MIMConfig:
    mask_ratio:    float = 0.60
    patch_size:    int   = 4
    loss:          str   = "l1"
    epochs:        int   = 50
    warmup_epochs: int   = 5
    base_lr:       float = 1e-4
    min_lr:        float = 1e-6
    weight_decay:  float = 0.05
    batch_size:    int   = 32
    peft_only:     bool  = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data:  DataConfig  = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    mim:   MIMConfig   = field(default_factory=MIMConfig)

    def __post_init__(self):
        self.model.num_classes = self.data.num_classes
        self.model.img_size    = self.data.img_size


# =========================================================================== #
#  Pre-built dataset configs
# =========================================================================== #

def get_dataset701_config(data_root: str = "./data/Dataset701_AbdomenCT") -> Config:
    cfg = Config()
    cfg.data.dataset      = "dataset701"
    cfg.data.num_classes  = 9
    cfg.data.img_size     = 224
    cfg.data.modality     = "ct"
    cfg.data.data_root    = data_root
    cfg.model.num_classes = 9
    cfg.model.img_size    = 224
    cfg.train.epochs      = 150
    cfg.train.base_lr     = 1e-4
    cfg.train.batch_size  = 16
    cfg.train.loss        = "ce_dice"
    cfg.train.experiment_name = "peft_umamba_dataset701_ct"
    return cfg


def get_dataset702_config(data_root: str = "./data/Dataset702_AbdomenMR") -> Config:
    cfg = Config()
    cfg.data.dataset       = "dataset702"
    cfg.data.num_classes   = 16
    cfg.data.img_size      = 224
    cfg.data.modality      = "mri"
    cfg.data.data_root     = data_root
    cfg.model.num_classes  = 16
    cfg.model.img_size     = 224
    cfg.train.epochs       = 200            # was 150; extra 50 epochs after encoder unfreeze
    # Optimizer: AdamW base; decoder gets higher LR via decoder_lr_scale
    cfg.train.optimizer    = "adamw"
    cfg.train.base_lr      = 1e-4
    cfg.train.weight_decay = 5e-3           # balanced regularisation
    cfg.train.supp_lr_scale    = 3.0
    cfg.train.decoder_lr_scale = 5.0        # was 10.0; less aggressive
    cfg.train.scheduler    = "poly"
    cfg.train.poly_power   = 0.9
    cfg.train.batch_size   = 24
    cfg.train.loss         = "ce_dice"
    cfg.train.dice_weight  = 0.5
    cfg.train.ce_weight    = 0.5
    cfg.train.bg_weight    = 0.1
    cfg.train.experiment_name = "peft_umamba_dataset702_mr"
    return cfg


def get_dataset704_config(data_root: str = "./data/Dataset704_Endovis17") -> Config:
    cfg = Config()
    cfg.data.dataset      = "dataset704"
    cfg.data.num_classes  = 8
    cfg.data.img_size     = 352
    cfg.data.modality     = "endoscopy"
    cfg.data.data_root    = data_root
    cfg.model.num_classes = 8
    cfg.model.img_size    = 352
    cfg.train.epochs      = 100
    cfg.train.base_lr     = 3e-4
    cfg.train.batch_size  = 8
    cfg.train.loss        = "focal_dice"
    cfg.train.experiment_name = "peft_umamba_dataset704_endovis"
    return cfg


def get_kvasir_config(data_root: str = "./data/kvasir") -> Config:
    cfg = Config()
    cfg.data.dataset      = "kvasir"
    cfg.data.num_classes  = 2
    cfg.data.img_size     = 352
    cfg.data.modality     = "endoscopy"
    cfg.data.data_root    = data_root
    cfg.model.num_classes = 2
    cfg.model.img_size    = 352
    cfg.train.epochs      = 100
    cfg.train.base_lr     = 3e-4
    cfg.train.batch_size  = 8
    cfg.train.loss        = "focal_dice"
    cfg.train.experiment_name = "peft_umamba_kvasir"
    return cfg


# Aliases
get_synapse_config   = get_dataset701_config
get_amos_config      = get_dataset702_config
get_endovis_config   = get_dataset704_config
get_clinicdb_config  = get_kvasir_config
get_endoscopy_config = get_kvasir_config


def config_from_dataset_name(name: str, data_root: str = None) -> Config:
    from data.dataset import resolve_dataset_key
    key = resolve_dataset_key(name)
    builders = {
        "dataset701": get_dataset701_config,
        "dataset702": get_dataset702_config,
        "dataset704": get_dataset704_config,
        "kvasir":     get_kvasir_config,
    }
    return builders[key](data_root) if data_root else builders[key]()
 """
"""
configs/config.py - Hyperparameter configuration for PEFT-UMamba.

Key changes from previous version

   gradient_clip  5.0  1.0   (prevent large early-training gradients)
   warmup_epochs    10  5     (faster PEFT ramp-up)
   eval_interval     1  5     (volumetric eval is expensive; skip early epochs)
   save_interval    10  5     (catch best model more frequently)
   Separate LR groups in build_optimizer (see utils.py):
      supp params  : lr  3  (need to learn fast from frozen backbone)
      decoder      : lr  1
      head         : lr  1
"""
""" 
from dataclasses import dataclass, field
from typing import List, Optional


# =========================================================================== #
#  Sub-configs
# =========================================================================== #

@dataclass
class SSMConfig:
    d_state:       int   = 16       # base SSM state dimension K
    d_state_supp:  int   = 4        # supplementary dimension K
    d_conv:        int   = 4
    expand:        int   = 2
    dt_min:        float = 0.001
    dt_max:        float = 0.1
    dt_init_floor: float = 1e-4


@dataclass
class PEFTConfig:
    use_supplementary_scan: bool  = True
    supp_state_dim:         int   = 4       # K
    ni_noise_std:           float = 1e-6

    use_lora:     bool  = False
    lora_rank:    int   = 4
    lora_alpha:   float = 8.0
    lora_dropout: float = 0.0

    supp_init:    str   = "neighbourhood"  # NI ablation: neighbourhood|zero|random_normal|xavier|copy_frozen
    use_sdlora:   bool  = False            # SDLoRA ablation

    train_supplementary: bool = True
    train_head:          bool = True
    train_decoder_norm:  bool = True
    train_patch_expand:  bool = True


@dataclass
class ModelConfig:
    backbone:    str       = "vmamba_tiny"
    img_size:    int       = 224
    in_channels: int       = 3
    num_classes: int       = 9

    depths:    List[int] = field(default_factory=lambda: [2, 2, 9, 2])
    feat_dims: List[int] = field(default_factory=lambda: [96, 192, 384, 768])

    pretrained_path: Optional[str] = None
    freeze_encoder:  bool          = True

    ssm:  SSMConfig  = field(default_factory=SSMConfig)
    peft: PEFTConfig = field(default_factory=PEFTConfig)


@dataclass
class DataConfig:
    dataset:         str   = "dataset702"
    data_root:       str   = "./data"
    img_size:        int   = 224
    num_classes:     int   = 16
    modality:        str   = "mri"
    train_val_split: float = 0.85
    num_workers:     int   = 4


@dataclass
class TrainConfig:
    stage:         str   = "finetune"
    epochs:        int   = 150
    warmup_epochs: int   = 5

    optimizer:     str   = "adamw"
    base_lr:       float = 1e-4
    min_lr:        float = 1e-6
    weight_decay:  float = 1e-4
    gradient_clip: float = 1.0

    supp_lr_scale:    float = 3.0    # supplementary SSM params: lr  3
    decoder_lr_scale: float = 10.0   # decoder (random init): lr  10
    head_lr_scale:    float = 10.0

    scheduler:    str   = "poly"
    poly_power:   float = 0.9

    loss:         str   = "ce_dice"
    dice_weight:  float = 0.5
    ce_weight:    float = 0.5
    focal_gamma:  float = 2.0
    bg_weight:    float = 0.1

    batch_size:         int = 24
    accumulation_steps: int = 1

    log_interval:  int = 50
    eval_interval: int = 1
    save_interval: int = 5
    output_dir:    str = "./outputs"
    experiment_name: str = "peft_umamba"

    use_amp:     bool = True
    distributed: bool = False
    seed:        int  = 42


@dataclass
class MIMConfig:
    mask_ratio:    float = 0.60
    patch_size:    int   = 4
    loss:          str   = "l1"
    epochs:        int   = 50
    warmup_epochs: int   = 5
    base_lr:       float = 1e-4
    min_lr:        float = 1e-6
    weight_decay:  float = 0.05
    batch_size:    int   = 32
    peft_only:     bool  = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data:  DataConfig  = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    mim:   MIMConfig   = field(default_factory=MIMConfig)

    def __post_init__(self):
        self.model.num_classes = self.data.num_classes
        self.model.img_size    = self.data.img_size


# =========================================================================== #
#  Pre-built dataset configs
# =========================================================================== #

def get_dataset701_config(data_root: str = "./data/Dataset701_AbdomenCT") -> Config:
    cfg = Config()
    cfg.data.dataset      = "dataset701"
    cfg.data.num_classes  = 9
    cfg.data.img_size     = 224
    cfg.data.modality     = "ct"
    cfg.data.data_root    = data_root
    cfg.model.num_classes = 9
    cfg.model.img_size    = 224
    cfg.train.epochs      = 150
    cfg.train.base_lr     = 1e-4
    cfg.train.batch_size  = 16
    cfg.train.loss        = "ce_dice"
    cfg.train.experiment_name = "peft_umamba_dataset701_ct"
    return cfg


def get_dataset702_config(data_root: str = "./data/Dataset702_AbdomenMR") -> Config:
    cfg = Config()
    cfg.data.dataset       = "dataset702"
    cfg.data.num_classes   = 16
    cfg.data.img_size      = 224
    cfg.data.modality      = "mri"
    cfg.data.data_root     = data_root
    cfg.model.num_classes  = 16
    cfg.model.img_size     = 224
    cfg.train.epochs       = 300            # extended: resume ~ep90 + 200 more epochs
    # Optimizer: AdamW base; decoder gets higher LR via decoder_lr_scale
    cfg.train.optimizer    = "adamw"
    cfg.train.base_lr      = 1e-4
    cfg.train.weight_decay = 5e-3           # balanced regularisation
    cfg.train.supp_lr_scale    = 3.0
    cfg.train.decoder_lr_scale = 5.0        # was 10.0; less aggressive
    cfg.train.scheduler    = "poly"
    cfg.train.poly_power   = 0.9
    cfg.train.batch_size   = 24
    cfg.train.loss         = "ce_dice"
    cfg.train.dice_weight  = 0.5
    cfg.train.ce_weight    = 0.5
    cfg.train.bg_weight    = 0.1
    cfg.train.experiment_name = "peft_umamba_dataset702_mr"
    return cfg


def get_dataset704_config(data_root: str = "./data/Dataset704_Endovis17") -> Config:
    cfg = Config()
    cfg.data.dataset      = "dataset704"
    cfg.data.num_classes  = 8
    cfg.data.img_size     = 352
    cfg.data.modality     = "endoscopy"
    cfg.data.data_root    = data_root
    cfg.model.num_classes = 8
    cfg.model.img_size    = 352
    cfg.train.epochs      = 100
    cfg.train.base_lr     = 3e-4
    cfg.train.batch_size  = 8
    cfg.train.loss        = "focal_dice"
    cfg.train.experiment_name = "peft_umamba_dataset704_endovis"
    return cfg


def get_kvasir_config(data_root: str = "./data/kvasir") -> Config:
    cfg = Config()
    cfg.data.dataset      = "kvasir"
    cfg.data.num_classes  = 2
    cfg.data.img_size     = 352
    cfg.data.modality     = "endoscopy"
    cfg.data.data_root    = data_root
    cfg.model.num_classes = 2
    cfg.model.img_size    = 352
    cfg.train.epochs      = 100
    cfg.train.base_lr     = 3e-4
    cfg.train.batch_size  = 8
    cfg.train.loss        = "focal_dice"
    cfg.train.experiment_name = "peft_umamba_kvasir"
    return cfg


# Aliases
get_synapse_config   = get_dataset701_config
get_amos_config      = get_dataset702_config
get_endovis_config   = get_dataset704_config
get_clinicdb_config  = get_kvasir_config
get_endoscopy_config = get_kvasir_config


def config_from_dataset_name(name: str, data_root: str = None) -> Config:
    from data.dataset import resolve_dataset_key
    key = resolve_dataset_key(name)
    builders = {
        "dataset701": get_dataset701_config,
        "dataset702": get_dataset702_config,
        "dataset704": get_dataset704_config,
        "kvasir":     get_kvasir_config,
    }
    return builders[key](data_root) if data_root else builders[key]() """
"""
configs/config.py - Hyperparameter configuration for PEFT-UMamba.

Key changes from previous version

   gradient_clip  5.0  1.0   (prevent large early-training gradients)
   warmup_epochs    10  5     (faster PEFT ramp-up)
   eval_interval     1  5     (volumetric eval is expensive; skip early epochs)
   save_interval    10  5     (catch best model more frequently)
   Separate LR groups in build_optimizer (see utils.py):
      supp params  : lr  3  (need to learn fast from frozen backbone)
      decoder      : lr  1
      head         : lr  1
"""

from dataclasses import dataclass, field
from typing import List, Optional


# =========================================================================== #
#  Sub-configs
# =========================================================================== #

@dataclass
class SSMConfig:
    d_state:       int   = 16       # base SSM state dimension K
    d_state_supp:  int   = 4        # supplementary dimension K
    d_conv:        int   = 4
    expand:        int   = 2
    dt_min:        float = 0.001
    dt_max:        float = 0.1
    dt_init_floor: float = 1e-4


@dataclass
class PEFTConfig:
    use_supplementary_scan: bool  = True
    supp_state_dim:         int   = 4       # K
    ni_noise_std:           float = 1e-6

    use_lora:     bool  = False
    lora_rank:    int   = 4
    lora_alpha:   float = 8.0
    lora_dropout: float = 0.0

    supp_init:    str   = "neighbourhood"  # NI ablation: neighbourhood|zero|random_normal|xavier|copy_frozen
    use_sdlora:   bool  = False            # SDLoRA ablation
    no_skip_gate: bool  = False            # Skip Attention Gate ablation

    train_supplementary: bool = True
    train_head:          bool = True
    train_decoder_norm:  bool = True
    train_patch_expand:  bool = True


@dataclass
class ModelConfig:
    backbone:    str       = "vmamba_tiny"
    img_size:    int       = 224
    in_channels: int       = 3
    num_classes: int       = 9

    depths:    List[int] = field(default_factory=lambda: [2, 2, 9, 2])
    feat_dims: List[int] = field(default_factory=lambda: [96, 192, 384, 768])

    pretrained_path: Optional[str] = None
    freeze_encoder:  bool          = True

    ssm:  SSMConfig  = field(default_factory=SSMConfig)
    peft: PEFTConfig = field(default_factory=PEFTConfig)


@dataclass
class DataConfig:
    dataset:         str   = "dataset702"
    data_root:       str   = "./data"
    img_size:        int   = 224
    num_classes:     int   = 16
    modality:        str   = "mri"
    train_val_split: float = 0.85
    num_workers:     int   = 4


@dataclass
class TrainConfig:
    stage:         str   = "finetune"
    epochs:        int   = 150
    warmup_epochs: int   = 5

    optimizer:     str   = "adamw"
    base_lr:       float = 1e-4
    min_lr:        float = 1e-6
    weight_decay:  float = 1e-4
    gradient_clip: float = 1.0

    supp_lr_scale:    float = 3.0    # supplementary SSM params: lr  3
    decoder_lr_scale: float = 10.0   # decoder (random init): lr  10
    head_lr_scale:    float = 10.0

    scheduler:    str   = "poly"
    poly_power:   float = 0.9

    loss:         str   = "ce_dice"
    dice_weight:  float = 0.5
    ce_weight:    float = 0.5
    focal_gamma:  float = 2.0
    bg_weight:    float = 0.1

    batch_size:         int = 24
    accumulation_steps: int = 1

    log_interval:  int = 50
    eval_interval: int = 1
    save_interval: int = 5
    output_dir:    str = "./outputs"
    experiment_name: str = "peft_umamba"

    use_amp:     bool = True
    distributed: bool = False
    seed:        int  = 42


@dataclass
class MIMConfig:
    mask_ratio:    float = 0.60
    patch_size:    int   = 4
    loss:          str   = "l1"
    epochs:        int   = 50
    warmup_epochs: int   = 5
    base_lr:       float = 1e-4
    min_lr:        float = 1e-6
    weight_decay:  float = 0.05
    batch_size:    int   = 32
    peft_only:     bool  = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data:  DataConfig  = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    mim:   MIMConfig   = field(default_factory=MIMConfig)

    def __post_init__(self):
        self.model.num_classes = self.data.num_classes
        self.model.img_size    = self.data.img_size


# =========================================================================== #
#  Pre-built dataset configs
# =========================================================================== #

def get_dataset701_config(data_root: str = "./data/Dataset701_AbdomenCT") -> Config:
    """Dataset701_AbdomenCT - 9-class CT (same layout as Synapse)."""
    cfg = Config()
    cfg.data.dataset      = "dataset701"
    cfg.data.num_classes  = 9
    cfg.data.img_size     = 224
    cfg.data.modality     = "ct"
    cfg.data.data_root    = data_root
    cfg.model.num_classes = 9
    cfg.model.img_size    = 224
    cfg.train.epochs      = 150
    cfg.train.base_lr     = 1e-4
    cfg.train.batch_size  = 16
    cfg.train.loss        = "ce_dice"
    cfg.train.experiment_name = "peft_umamba_dataset701_ct"
    return cfg


def get_dataset702_config(data_root: str = "./data/Dataset702_AbdomenMR") -> Config:
    """Dataset702_AbdomenMR - 16-class MRI (AMOS22)."""
    cfg = Config()
    cfg.data.dataset       = "dataset702"
    cfg.data.num_classes   = 16
    cfg.data.img_size      = 224
    cfg.data.modality      = "mri"
    cfg.data.data_root     = data_root
    cfg.model.num_classes  = 16
    cfg.model.img_size     = 224
    cfg.train.epochs       = 300            # extended: resume ~ep90 + 200 more epochs
    # Optimizer: AdamW base; decoder gets higher LR via decoder_lr_scale
    cfg.train.optimizer    = "adamw"
    cfg.train.base_lr      = 1e-4
    cfg.train.weight_decay = 5e-3           # balanced regularisation
    cfg.train.supp_lr_scale    = 3.0
    cfg.train.decoder_lr_scale = 5.0        # was 10.0; less aggressive
    cfg.train.scheduler    = "poly"
    cfg.train.poly_power   = 0.9
    cfg.train.batch_size   = 24
    cfg.train.loss         = "ce_dice"
    cfg.train.dice_weight  = 0.5
    cfg.train.ce_weight    = 0.5
    cfg.train.bg_weight    = 0.1
    cfg.train.experiment_name = "peft_umamba_dataset702_mr"
    return cfg


def get_dataset704_config(data_root: str = "./data/Dataset704_Endovis17") -> Config:
    """Dataset704_Endovis17 - 2-class surgical instrument."""
    cfg = Config()
    cfg.data.dataset      = "dataset704"
    cfg.data.num_classes  = 8
    cfg.data.img_size     = 352
    cfg.data.modality     = "endoscopy"
    cfg.data.data_root    = data_root
    cfg.model.num_classes = 8
    cfg.model.img_size    = 352
    cfg.train.epochs      = 100
    cfg.train.base_lr     = 3e-4
    cfg.train.batch_size  = 8
    cfg.train.loss        = "focal_dice"
    cfg.train.experiment_name = "peft_umamba_dataset704_endovis"
    return cfg


def get_kvasir_config(data_root: str = "./data/kvasir") -> Config:
    """Kvasir-SEG - 2-class polyp segmentation."""
    cfg = Config()
    cfg.data.dataset      = "kvasir"
    cfg.data.num_classes  = 2
    cfg.data.img_size     = 352
    cfg.data.modality     = "endoscopy"
    cfg.data.data_root    = data_root
    cfg.model.num_classes = 2
    cfg.model.img_size    = 352
    cfg.train.epochs      = 100
    cfg.train.base_lr     = 3e-4
    cfg.train.batch_size  = 8
    cfg.train.loss        = "focal_dice"
    cfg.train.experiment_name = "peft_umamba_kvasir"
    return cfg


def get_dataset703_config(data_root: str = "./data/Dataset703_NeurIPSCell") -> Config:
    """NeurIPS 2022 Cell Segmentation - 3-class (background/interior/boundary)."""
    cfg = Config()
    cfg.data.dataset      = "dataset703"
    cfg.data.num_classes  = 3
    cfg.data.img_size     = 352
    cfg.data.modality     = "microscopy"
    cfg.data.data_root    = data_root
    cfg.model.num_classes = 3
    cfg.model.img_size    = 352
    cfg.train.epochs      = 150
    cfg.train.base_lr     = 3e-4
    cfg.train.batch_size  = 8
    cfg.train.loss        = "ce_dice"
    cfg.train.experiment_name = "peft_umamba_neurips_cell"
    return cfg


# Aliases
get_synapse_config   = get_dataset701_config
get_amos_config      = get_dataset702_config
get_endovis_config   = get_dataset704_config
get_clinicdb_config  = get_kvasir_config
get_endoscopy_config = get_kvasir_config
get_cell_config      = get_dataset703_config


def config_from_dataset_name(name: str, data_root: str = None) -> Config:
    from data.dataset import resolve_dataset_key
    key = resolve_dataset_key(name)
    builders = {
        "dataset701": get_dataset701_config,
        "dataset702": get_dataset702_config,
        "dataset703": get_dataset703_config,
        "dataset704": get_dataset704_config,
        "kvasir":     get_kvasir_config,
    }
    return builders[key](data_root) if data_root else builders[key]()
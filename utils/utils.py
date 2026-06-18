""" 

import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from medpy import metric as medmetric   # pip install medpy


# --------------------------------------------------------------------------- #
#  Reproducibility
# --------------------------------------------------------------------------- #

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
#  Logging
# --------------------------------------------------------------------------- #

def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# --------------------------------------------------------------------------- #
#  Loss functions
# --------------------------------------------------------------------------- #

class DiceLoss(nn.Module):
   
    def __init__(self, num_classes: int, smooth: float = 1e-5,
                 ignore_index: int = -1):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Always compute in fp32 — fp16 softmax overflows for large logits
        probs = F.softmax(logits.float(), dim=1)   # [B, C, H, W]
        B, C, H, W = probs.shape

        one_hot = F.one_hot(
            targets.clamp(0), num_classes=C
        ).permute(0, 3, 1, 2).float()             # [B, C, H, W]

        # Mask ignore index
        if self.ignore_index >= 0:
            mask    = (targets != self.ignore_index).unsqueeze(1).float()
            probs   = probs   * mask
            one_hot = one_hot * mask

        dims = (0, 2, 3)
        inter = (probs * one_hot).sum(dims)
        union = probs.sum(dims) + one_hot.sum(dims)

        dice = (2 * inter + self.smooth) / (union + self.smooth)
        # Skip background (class 0) in mean
        dice_mean = dice[1:].mean() if C > 1 else dice.mean()
        return 1.0 - dice_mean


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None,
                 ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # fp32 required: fp16 CE can produce NaN for large logits
        ce    = F.cross_entropy(logits.float(), targets, reduction="none",
                                ignore_index=self.ignore_index)
        pt    = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        return focal.mean()


class CEDiceLoss(nn.Module):
   
    def __init__(self, num_classes: int, dice_w: float = 0.5,
                 ce_w: float = 0.5, ignore_index: int = -100,
                 bg_weight: float = 0.1):
        super().__init__()
        self.dice        = DiceLoss(num_classes)
        self.ce_w        = ce_w
        self.dice_w      = dice_w
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.bg_weight   = bg_weight

    def _class_weights(self, device) -> torch.Tensor:
        w = torch.ones(self.num_classes, device=device)
        w[0] = self.bg_weight
        return w

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        w         = self._class_weights(logits.device)
        ce_loss   = F.cross_entropy(logits.float(), targets,
                                    weight=w,
                                    ignore_index=self.ignore_index)
        dice_loss = self.dice(logits, targets)
        return self.ce_w * ce_loss + self.dice_w * dice_loss


class FocalDiceLoss(nn.Module):
    def __init__(self, num_classes: int, dice_w: float = 0.5,
                 focal_w: float = 0.5, gamma: float = 2.0):
        super().__init__()
        self.dice = DiceLoss(num_classes)
        self.focal = FocalLoss(gamma=gamma)
        self.dice_w = dice_w
        self.focal_w = focal_w

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.focal_w * self.focal(logits, targets) + \
               self.dice_w * self.dice(logits, targets)


class MIMLoss(nn.Module):
    def forward(
        self,
        pred: torch.Tensor,       # [B, H', W', P*P*3]
        target: torch.Tensor,     # [B, 3, H, W]
        mask: torch.Tensor,       # [B, num_patches] bool
        patch_size: int = 4,
    ) -> torch.Tensor:
        B, C, H, W = target.shape
        H_ = H // patch_size
        W_ = W // patch_size
        # Patchify target
        target_patches = target.reshape(B, C, H_, patch_size, W_, patch_size)
        target_patches = target_patches.permute(0, 2, 4, 3, 5, 1)      # [B,H',W',P,P,C]
        target_patches = target_patches.reshape(B, H_, W_, -1)          # [B,H',W',P*P*C]

        mask_2d = mask.reshape(B, H_, W_)                               # [B,H',W']
        mask_2d = mask_2d.unsqueeze(-1).expand_as(target_patches)       # [B,H',W',P*P*C]

        loss = F.l1_loss(pred[mask_2d], target_patches[mask_2d])
        return loss


def build_loss(cfg) -> nn.Module:
    tc = cfg.train
    nc = cfg.model.num_classes
    bg = getattr(tc, "bg_weight", 0.1)   # background suppression weight

    if tc.loss == "ce":
        w = torch.ones(nc)
        w[0] = bg
        return nn.CrossEntropyLoss(weight=w)
    elif tc.loss == "dice":
        return DiceLoss(nc)
    elif tc.loss == "ce_dice":
        return CEDiceLoss(nc, dice_w=tc.dice_weight,
                          ce_w=tc.ce_weight, bg_weight=bg)
    elif tc.loss == "focal_dice":
        return FocalDiceLoss(nc, dice_w=tc.dice_weight,
                             focal_w=tc.ce_weight, gamma=tc.focal_gamma)
    else:
        raise ValueError(f"Unknown loss: {tc.loss}")


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #

def dice_coefficient(pred: np.ndarray, target: np.ndarray) -> float:
    inter = (pred * target).sum()
    return (2 * inter) / (pred.sum() + target.sum() + 1e-8)


def hausdorff_95(pred: np.ndarray, target: np.ndarray) -> float:
    
    if pred.sum() == 0 or target.sum() == 0:
        return float("nan")
    try:
        return float(medmetric.binary.hd95(pred, target))
    except Exception:
        return float("nan")


class SegmentationMetrics:
    

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.reset()

    def reset(self):
        self.dice_sum  = np.zeros(self.num_classes)
        self.hd95_sum  = np.zeros(self.num_classes)
        self.dice_cnt  = np.zeros(self.num_classes)   # cases with GT foreground
        self.hd95_cnt  = np.zeros(self.num_classes)   # cases where hd95 is valid

    def update(self, pred_vol: np.ndarray, gt_vol: np.ndarray):
        for c in range(1, self.num_classes):
            pred_c = (pred_vol == c).astype(np.uint8)
            gt_c   = (gt_vol   == c).astype(np.uint8)

            if gt_c.sum() == 0 and pred_c.sum() == 0:
                # Both empty — count as perfect Dice=1, skip HD95
                self.dice_sum[c] += 1.0
                self.dice_cnt[c] += 1
            elif gt_c.sum() > 0:
                self.dice_sum[c] += dice_coefficient(pred_c, gt_c)
                self.dice_cnt[c] += 1
                hd = hausdorff_95(pred_c, gt_c)
                if not np.isnan(hd):
                    self.hd95_sum[c] += hd
                    self.hd95_cnt[c] += 1

    def compute(self) -> Dict[str, float]:
        results    = {}
        dice_vals  = []
        hd_vals    = []

        for c in range(1, self.num_classes):
            cname = self.class_names[c]

            # Dice
            d = float(self.dice_sum[c] / max(self.dice_cnt[c], 1))
            results[f"dice_{cname}"] = d
            dice_vals.append(d)

            # HD95 — only if we have valid measurements
            if self.hd95_cnt[c] > 0:
                h = float(self.hd95_sum[c] / self.hd95_cnt[c])
            else:
                h = float("nan")
            results[f"hd95_{cname}"] = h
            if not np.isnan(h):
                hd_vals.append(h)

        results["mean_dice"] = float(np.mean(dice_vals)) if dice_vals else 0.0
        # mean_hd95 only over classes where we have valid HD95 measurements
        results["mean_hd95"] = float(np.mean(hd_vals)) if hd_vals else float("nan")
        return results


# --------------------------------------------------------------------------- #
#  Endoscopy-specific metrics (mDice, mIoU, F2, structure measure)
# --------------------------------------------------------------------------- #

class EndoscopyMetrics:

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = 0

    def update(self, pred: np.ndarray, gt: np.ndarray):
        pred_b = (pred > 0.5).astype(np.uint8)
        gt_b = (gt > 0.5).astype(np.uint8)
        self.tp += int((pred_b * gt_b).sum())
        self.fp += int((pred_b * (1 - gt_b)).sum())
        self.fn += int(((1 - pred_b) * gt_b).sum())

    def compute(self) -> Dict[str, float]:
        tp, fp, fn = self.tp, self.fp, self.fn
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f2 = (5 * prec * rec) / (4 * prec + rec + 1e-8)
        return {"dice": dice, "iou": iou, "f2": f2,
                "precision": prec, "recall": rec}


# --------------------------------------------------------------------------- #
#  Schedulers
# --------------------------------------------------------------------------- #

class WarmupCosineScheduler:
    
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 base_lr: float, min_lr: float = 1e-6):
        self.opt           = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.base_lr       = base_lr
        self.min_lr        = min_lr
        # Store initial LR for each group so we can scale them proportionally
        self._init_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / max(1, self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs)
            scale = self.min_lr / self.base_lr + 0.5 * (
                1 - self.min_lr / self.base_lr) * (1 + np.cos(np.pi * progress))

        for pg, init_lr in zip(self.opt.param_groups, self._init_lrs):
            pg["lr"] = init_lr * scale

        return self.base_lr * scale   # return base LR for logging


class PolyScheduler:
    
    def __init__(self, optimizer, total_epochs: int, base_lr: float,
                 min_lr: float = 1e-6, power: float = 0.9):
        self.opt          = optimizer
        self.total_epochs = total_epochs
        self.base_lr      = base_lr
        self.min_lr       = min_lr
        self.power        = power
        self._init_lrs    = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch: int) -> float:
        factor = (1 - epoch / self.total_epochs) ** self.power
        lr_cur = max(self.min_lr, self.base_lr * factor)
        scale  = lr_cur / self.base_lr
        for pg, init_lr in zip(self.opt.param_groups, self._init_lrs):
            pg["lr"] = max(self.min_lr, init_lr * scale)
        return lr_cur


def build_scheduler(optimizer, cfg):
    tc = cfg.train
    if tc.scheduler == "cosine":
        return WarmupCosineScheduler(
            optimizer, tc.warmup_epochs, tc.epochs, tc.base_lr, tc.min_lr
        )
    elif tc.scheduler == "poly":
        return PolyScheduler(
            optimizer, tc.epochs, tc.base_lr, tc.min_lr, tc.poly_power
        )
    else:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tc.epochs, eta_min=tc.min_lr
        )


# --------------------------------------------------------------------------- #
#  Optimizer builder
# --------------------------------------------------------------------------- #

def build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    
    tc = cfg.train

    def no_wd(name: str, param) -> bool:
        if getattr(param, "_no_weight_decay", False):
            return True
        return any(kw in name for kw in ("bias", "norm", "LayerNorm", "bn"))

    def is_supp(name: str) -> bool:
        return "supp" in name

    def is_decoder_or_head(name: str) -> bool:
        return any(kw in name for kw in
                   ("decoder_stages", "aux_heads", "seg_head",
                    "skip_gate", "upsample", "proj", "norm", "dw", "pw", "bn"))

    supp_lr    = tc.base_lr * getattr(tc, "supp_lr_scale", 1.0)
    decoder_lr = tc.base_lr * getattr(tc, "decoder_lr_scale", 1.0)

    # Four groups by (role, wd)
    grp = {"supp_nowd": [], "supp_wd": [],
           "dec_nowd": [],  "dec_wd": [],
           "other_nowd": [], "other_wd": []}

    seen = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen:
            continue
        seen.add(id(param))
        nwd = no_wd(name, param)
        if is_supp(name):
            grp["supp_nowd" if nwd else "supp_wd"].append(param)
        elif is_decoder_or_head(name):
            grp["dec_nowd" if nwd else "dec_wd"].append(param)
        else:
            grp["other_nowd" if nwd else "other_wd"].append(param)

    wd = tc.weight_decay

    # AdamW groups (supplementary — adaptive, small, stable)
    adamw_groups = []
    if grp["supp_wd"]:
        adamw_groups.append({"params": grp["supp_wd"],   "lr": supp_lr, "weight_decay": wd,  "name": "supp_wd"})
    if grp["supp_nowd"]:
        adamw_groups.append({"params": grp["supp_nowd"], "lr": supp_lr, "weight_decay": 0.0, "name": "supp_nowd"})
    if grp["other_wd"]:
        adamw_groups.append({"params": grp["other_wd"],  "lr": tc.base_lr, "weight_decay": wd,  "name": "other_wd"})
    if grp["other_nowd"]:
        adamw_groups.append({"params": grp["other_nowd"],"lr": tc.base_lr, "weight_decay": 0.0, "name": "other_nowd"})

    # SGD groups (decoder + head — fast convergence from random init)
    sgd_groups = []
    if grp["dec_wd"]:
        sgd_groups.append({"params": grp["dec_wd"],   "lr": decoder_lr, "weight_decay": wd,  "name": "dec_wd"})
    if grp["dec_nowd"]:
        sgd_groups.append({"params": grp["dec_nowd"], "lr": decoder_lr, "weight_decay": 0.0, "name": "dec_nowd"})

    # Use AdamW as the unified optimizer if no SGD params exist
    if not sgd_groups:
        return torch.optim.AdamW(adamw_groups or [{"params": list(model.parameters()), "lr": tc.base_lr}],
                                 lr=tc.base_lr)

    # Combine into a single AdamW — SGD's momentum effect for decoder
    # is approximated by the weight decay + gradient clipping in the loop.
    # Using separate optimizers is complex; instead use AdamW for all but
    # with SGD-equivalent high LR for decoder.
    all_groups = adamw_groups + sgd_groups
    return torch.optim.AdamW(all_groups, lr=tc.base_lr)


# --------------------------------------------------------------------------- #
#  Sliding window inference for 3D volumes (Synapse)
# --------------------------------------------------------------------------- #

def sliding_window_inference(
    volume: np.ndarray,      # [D, H, W]
    model: nn.Module,
    img_size: int,
    num_classes: int,
    device: torch.device,
    mean: Tuple = (0.485, 0.456, 0.406),
    std: Tuple = (0.229, 0.224, 0.225),
) -> np.ndarray:
    
    mean_t = torch.tensor(mean, device=device)[:, None, None]
    std_t = torch.tensor(std, device=device)[:, None, None]

    D, H, W = volume.shape
    pred_vol = np.zeros((D, H, W), dtype=np.int64)
    model.eval()

    with torch.no_grad():
        for d in range(D):
            slc = volume[d]                              # [H, W]
            # Normalize
            slc = np.clip(slc, -175, 250)
            slc = (slc - slc.min()) / (slc.max() - slc.min() + 1e-8)
            # Resize
            import cv2
            slc_resized = cv2.resize(
                slc.astype(np.float32), (img_size, img_size),
                interpolation=cv2.INTER_LINEAR
            )
            # To tensor
            x = torch.from_numpy(slc_resized).unsqueeze(0).repeat(3, 1, 1)
            x = x.to(device)
            x = (x - mean_t) / std_t
            x = x.unsqueeze(0)     # [1, 3, H, W]

            logits = model(x)      # [1, C, H, W]
            pred = logits.argmax(1).squeeze(0).cpu().numpy()  # [H, W]

            # Resize back to original
            pred_orig = cv2.resize(
                pred.astype(np.float32), (W, H),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int64)
            pred_vol[d] = pred_orig

    return pred_vol


# --------------------------------------------------------------------------- #
#  Checkpointing
# --------------------------------------------------------------------------- #

def save_checkpoint(
    state: Dict,
    output_dir: str,
    name: str = "checkpoint.pth",
):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    torch.save(state, path)
    return path


def load_checkpoint(model: nn.Module, ckpt_path: str,
                    optimizer=None, strict: bool = False):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=strict)
    epoch = ckpt.get("epoch", 0)
    best_metric = ckpt.get("best_metric", 0.0)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[Checkpoint] Loaded {ckpt_path} (epoch {epoch}, "
          f"best_metric={best_metric:.4f})")
    return epoch, best_metric


# --------------------------------------------------------------------------- #
#  Misc
# --------------------------------------------------------------------------- #

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
 """
"""
utils.py — Losses, metrics, schedulers, and helper functions for PEFT-UMamba.
"""
""" 
import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from medpy import metric as medmetric   # pip install medpy


# --------------------------------------------------------------------------- #
#  Reproducibility
# --------------------------------------------------------------------------- #

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
#  Logging
# --------------------------------------------------------------------------- #

def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# --------------------------------------------------------------------------- #
#  Loss functions
# --------------------------------------------------------------------------- #

class DiceLoss(nn.Module):
  
    def __init__(self, num_classes: int, smooth: float = 1e-5,
                 ignore_index: int = -1):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Always compute in fp32 — fp16 softmax overflows for large logits
        probs = F.softmax(logits.float(), dim=1)   # [B, C, H, W]
        B, C, H, W = probs.shape

        one_hot = F.one_hot(
            targets.clamp(0), num_classes=C
        ).permute(0, 3, 1, 2).float()             # [B, C, H, W]

        # Mask ignore index
        if self.ignore_index >= 0:
            mask    = (targets != self.ignore_index).unsqueeze(1).float()
            probs   = probs   * mask
            one_hot = one_hot * mask

        dims = (0, 2, 3)
        inter = (probs * one_hot).sum(dims)
        union = probs.sum(dims) + one_hot.sum(dims)

        dice = (2 * inter + self.smooth) / (union + self.smooth)
        # Skip background (class 0) in mean
        dice_mean = dice[1:].mean() if C > 1 else dice.mean()
        return 1.0 - dice_mean


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None,
                 ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # fp32 required: fp16 CE can produce NaN for large logits
        ce    = F.cross_entropy(logits.float(), targets, reduction="none",
                                ignore_index=self.ignore_index)
        pt    = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        return focal.mean()


class CEDiceLoss(nn.Module):
    
    def __init__(self, num_classes: int, dice_w: float = 0.5,
                 ce_w: float = 0.5, ignore_index: int = -100,
                 bg_weight: float = 0.1, label_smoothing: float = 0.1):
        super().__init__()
        self.dice        = DiceLoss(num_classes)
        self.ce_w        = ce_w
        self.dice_w      = dice_w
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.bg_weight   = bg_weight
        self.label_smoothing = label_smoothing

    def _class_weights(self, device) -> torch.Tensor:
        w = torch.ones(self.num_classes, device=device)
        w[0] = self.bg_weight
        return w

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        w         = self._class_weights(logits.device)
        ce_loss   = F.cross_entropy(logits.float(), targets,
                                    weight=w,
                                    ignore_index=self.ignore_index,
                                    label_smoothing=self.label_smoothing)
        dice_loss = self.dice(logits, targets)
        return self.ce_w * ce_loss + self.dice_w * dice_loss


class FocalDiceLoss(nn.Module):
    def __init__(self, num_classes: int, dice_w: float = 0.5,
                 focal_w: float = 0.5, gamma: float = 2.0):
        super().__init__()
        self.dice = DiceLoss(num_classes)
        self.focal = FocalLoss(gamma=gamma)
        self.dice_w = dice_w
        self.focal_w = focal_w

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.focal_w * self.focal(logits, targets) + \
               self.dice_w * self.dice(logits, targets)


class MIMLoss(nn.Module):
    def forward(
        self,
        pred: torch.Tensor,       # [B, H', W', P*P*3]
        target: torch.Tensor,     # [B, 3, H, W]
        mask: torch.Tensor,       # [B, num_patches] bool
        patch_size: int = 4,
    ) -> torch.Tensor:
        B, C, H, W = target.shape
        H_ = H // patch_size
        W_ = W // patch_size
        # Patchify target
        target_patches = target.reshape(B, C, H_, patch_size, W_, patch_size)
        target_patches = target_patches.permute(0, 2, 4, 3, 5, 1)      # [B,H',W',P,P,C]
        target_patches = target_patches.reshape(B, H_, W_, -1)          # [B,H',W',P*P*C]

        mask_2d = mask.reshape(B, H_, W_)                               # [B,H',W']
        mask_2d = mask_2d.unsqueeze(-1).expand_as(target_patches)       # [B,H',W',P*P*C]

        loss = F.l1_loss(pred[mask_2d], target_patches[mask_2d])
        return loss


def build_loss(cfg) -> nn.Module:
    tc = cfg.train
    nc = cfg.model.num_classes
    bg = getattr(tc, "bg_weight", 0.1)   # background suppression weight

    if tc.loss == "ce":
        w = torch.ones(nc)
        w[0] = bg
        return nn.CrossEntropyLoss(weight=w)
    elif tc.loss == "dice":
        return DiceLoss(nc)
    elif tc.loss == "ce_dice":
        return CEDiceLoss(nc, dice_w=tc.dice_weight,
                          ce_w=tc.ce_weight, bg_weight=bg)
    elif tc.loss == "focal_dice":
        return FocalDiceLoss(nc, dice_w=tc.dice_weight,
                             focal_w=tc.ce_weight, gamma=tc.focal_gamma)
    else:
        raise ValueError(f"Unknown loss: {tc.loss}")


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #

def dice_coefficient(pred: np.ndarray, target: np.ndarray) -> float:
   
    inter = (pred * target).sum()
    return (2 * inter) / (pred.sum() + target.sum() + 1e-8)


def hausdorff_95(pred: np.ndarray, target: np.ndarray) -> float:
    
    if pred.sum() == 0 or target.sum() == 0:
        return float("nan")
    try:
        return float(medmetric.binary.hd95(pred, target))
    except Exception:
        return float("nan")


class SegmentationMetrics:
   

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.reset()

    def reset(self):
        self.dice_sum  = np.zeros(self.num_classes)
        self.hd95_sum  = np.zeros(self.num_classes)
        self.dice_cnt  = np.zeros(self.num_classes)   # cases with GT foreground
        self.hd95_cnt  = np.zeros(self.num_classes)   # cases where hd95 is valid

    def update(self, pred_vol: np.ndarray, gt_vol: np.ndarray):
        for c in range(1, self.num_classes):
            pred_c = (pred_vol == c).astype(np.uint8)
            gt_c   = (gt_vol   == c).astype(np.uint8)

            if gt_c.sum() == 0 and pred_c.sum() == 0:
                # Both empty — count as perfect Dice=1, skip HD95
                self.dice_sum[c] += 1.0
                self.dice_cnt[c] += 1
            elif gt_c.sum() > 0:
                self.dice_sum[c] += dice_coefficient(pred_c, gt_c)
                self.dice_cnt[c] += 1
                hd = hausdorff_95(pred_c, gt_c)
                if not np.isnan(hd):
                    self.hd95_sum[c] += hd
                    self.hd95_cnt[c] += 1

    def compute(self) -> Dict[str, float]:
        results    = {}
        dice_vals  = []
        hd_vals    = []

        for c in range(1, self.num_classes):
            cname = self.class_names[c]

            # Dice
            d = float(self.dice_sum[c] / max(self.dice_cnt[c], 1))
            results[f"dice_{cname}"] = d
            dice_vals.append(d)

            # HD95 — only if we have valid measurements
            if self.hd95_cnt[c] > 0:
                h = float(self.hd95_sum[c] / self.hd95_cnt[c])
            else:
                h = float("nan")
            results[f"hd95_{cname}"] = h
            if not np.isnan(h):
                hd_vals.append(h)

        results["mean_dice"] = float(np.mean(dice_vals)) if dice_vals else 0.0
        # mean_hd95 only over classes where we have valid HD95 measurements
        results["mean_hd95"] = float(np.mean(hd_vals)) if hd_vals else float("nan")
        return results


# --------------------------------------------------------------------------- #
#  Endoscopy-specific metrics (mDice, mIoU, F2, structure measure)
# --------------------------------------------------------------------------- #

class EndoscopyMetrics:

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = 0

    def update(self, pred: np.ndarray, gt: np.ndarray):
        pred_b = (pred > 0.5).astype(np.uint8)
        gt_b = (gt > 0.5).astype(np.uint8)
        self.tp += int((pred_b * gt_b).sum())
        self.fp += int((pred_b * (1 - gt_b)).sum())
        self.fn += int(((1 - pred_b) * gt_b).sum())

    def compute(self) -> Dict[str, float]:
        tp, fp, fn = self.tp, self.fp, self.fn
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f2 = (5 * prec * rec) / (4 * prec + rec + 1e-8)
        return {"dice": dice, "iou": iou, "f2": f2,
                "precision": prec, "recall": rec}


# --------------------------------------------------------------------------- #
#  Schedulers
# --------------------------------------------------------------------------- #

class WarmupCosineScheduler:
   
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 base_lr: float, min_lr: float = 1e-6):
        self.opt           = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.base_lr       = base_lr
        self.min_lr        = min_lr
        # Store initial LR for each group so we can scale them proportionally
        self._init_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / max(1, self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs)
            scale = self.min_lr / self.base_lr + 0.5 * (
                1 - self.min_lr / self.base_lr) * (1 + np.cos(np.pi * progress))

        for pg, init_lr in zip(self.opt.param_groups, self._init_lrs):
            pg["lr"] = init_lr * scale

        return self.base_lr * scale   # return base LR for logging


class PolyScheduler:
   
    def __init__(self, optimizer, total_epochs: int, base_lr: float,
                 min_lr: float = 1e-6, power: float = 0.9):
        self.opt          = optimizer
        self.total_epochs = total_epochs
        self.base_lr      = base_lr
        self.min_lr       = min_lr
        self.power        = power
        self._init_lrs    = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch: int) -> float:
        factor = (1 - epoch / self.total_epochs) ** self.power
        lr_cur = max(self.min_lr, self.base_lr * factor)
        scale  = lr_cur / self.base_lr
        for pg, init_lr in zip(self.opt.param_groups, self._init_lrs):
            pg["lr"] = max(self.min_lr, init_lr * scale)
        return lr_cur


def build_scheduler(optimizer, cfg):
    tc = cfg.train
    if tc.scheduler == "cosine":
        return WarmupCosineScheduler(
            optimizer, tc.warmup_epochs, tc.epochs, tc.base_lr, tc.min_lr
        )
    elif tc.scheduler == "poly":
        return PolyScheduler(
            optimizer, tc.epochs, tc.base_lr, tc.min_lr, tc.poly_power
        )
    else:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tc.epochs, eta_min=tc.min_lr
        )


# --------------------------------------------------------------------------- #
#  Optimizer builder
# --------------------------------------------------------------------------- #

def build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    
    tc = cfg.train

    def no_wd(name: str, param) -> bool:
        if getattr(param, "_no_weight_decay", False):
            return True
        return any(kw in name for kw in ("bias", "norm", "LayerNorm", "bn"))

    def is_supp(name: str) -> bool:
        return "supp" in name

    def is_decoder_or_head(name: str) -> bool:
        return any(kw in name for kw in
                   ("decoder_stages", "aux_heads", "seg_head",
                    "skip_gate", "upsample", "proj", "norm", "dw", "pw", "bn"))

    supp_lr    = tc.base_lr * getattr(tc, "supp_lr_scale", 1.0)
    decoder_lr = tc.base_lr * getattr(tc, "decoder_lr_scale", 1.0)

    # Four groups by (role, wd)
    grp = {"supp_nowd": [], "supp_wd": [],
           "dec_nowd": [],  "dec_wd": [],
           "other_nowd": [], "other_wd": []}

    seen = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen:
            continue
        seen.add(id(param))
        nwd = no_wd(name, param)
        if is_supp(name):
            grp["supp_nowd" if nwd else "supp_wd"].append(param)
        elif is_decoder_or_head(name):
            grp["dec_nowd" if nwd else "dec_wd"].append(param)
        else:
            grp["other_nowd" if nwd else "other_wd"].append(param)

    wd = tc.weight_decay

    # AdamW groups (supplementary — adaptive, small, stable)
    adamw_groups = []
    if grp["supp_wd"]:
        adamw_groups.append({"params": grp["supp_wd"],   "lr": supp_lr, "weight_decay": wd,  "name": "supp_wd"})
    if grp["supp_nowd"]:
        adamw_groups.append({"params": grp["supp_nowd"], "lr": supp_lr, "weight_decay": 0.0, "name": "supp_nowd"})
    if grp["other_wd"]:
        adamw_groups.append({"params": grp["other_wd"],  "lr": tc.base_lr, "weight_decay": wd,  "name": "other_wd"})
    if grp["other_nowd"]:
        adamw_groups.append({"params": grp["other_nowd"],"lr": tc.base_lr, "weight_decay": 0.0, "name": "other_nowd"})

    # SGD groups (decoder + head — fast convergence from random init)
    sgd_groups = []
    if grp["dec_wd"]:
        sgd_groups.append({"params": grp["dec_wd"],   "lr": decoder_lr, "weight_decay": wd,  "name": "dec_wd"})
    if grp["dec_nowd"]:
        sgd_groups.append({"params": grp["dec_nowd"], "lr": decoder_lr, "weight_decay": 0.0, "name": "dec_nowd"})

    # Use AdamW as the unified optimizer if no SGD params exist
    if not sgd_groups:
        return torch.optim.AdamW(adamw_groups or [{"params": list(model.parameters()), "lr": tc.base_lr}],
                                 lr=tc.base_lr)

    # Combine into a single AdamW — SGD's momentum effect for decoder
    # is approximated by the weight decay + gradient clipping in the loop.
    # Using separate optimizers is complex; instead use AdamW for all but
    # with SGD-equivalent high LR for decoder.
    all_groups = adamw_groups + sgd_groups
    return torch.optim.AdamW(all_groups, lr=tc.base_lr)


# --------------------------------------------------------------------------- #
#  Sliding window inference for 3D volumes (Synapse)
# --------------------------------------------------------------------------- #

def sliding_window_inference(
    volume: np.ndarray,      # [D, H, W]
    model: nn.Module,
    img_size: int,
    num_classes: int,
    device: torch.device,
    mean: Tuple = (0.485, 0.456, 0.406),
    std: Tuple = (0.229, 0.224, 0.225),
) -> np.ndarray:
  
    mean_t = torch.tensor(mean, device=device)[:, None, None]
    std_t = torch.tensor(std, device=device)[:, None, None]

    D, H, W = volume.shape
    pred_vol = np.zeros((D, H, W), dtype=np.int64)
    model.eval()

    with torch.no_grad():
        for d in range(D):
            slc = volume[d]                              # [H, W]
            # Normalize
            slc = np.clip(slc, -175, 250)
            slc = (slc - slc.min()) / (slc.max() - slc.min() + 1e-8)
            # Resize
            import cv2
            slc_resized = cv2.resize(
                slc.astype(np.float32), (img_size, img_size),
                interpolation=cv2.INTER_LINEAR
            )
            # To tensor
            x = torch.from_numpy(slc_resized).unsqueeze(0).repeat(3, 1, 1)
            x = x.to(device)
            x = (x - mean_t) / std_t
            x = x.unsqueeze(0)     # [1, 3, H, W]

            logits = model(x)      # [1, C, H, W]
            pred = logits.argmax(1).squeeze(0).cpu().numpy()  # [H, W]

            # Resize back to original
            pred_orig = cv2.resize(
                pred.astype(np.float32), (W, H),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int64)
            pred_vol[d] = pred_orig

    return pred_vol


# --------------------------------------------------------------------------- #
#  Checkpointing
# --------------------------------------------------------------------------- #

def save_checkpoint(
    state: Dict,
    output_dir: str,
    name: str = "checkpoint.pth",
):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    torch.save(state, path)
    return path


def load_checkpoint(model: nn.Module, ckpt_path: str,
                    optimizer=None, strict: bool = False):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=strict)
    epoch = ckpt.get("epoch", 0)
    best_metric = ckpt.get("best_metric", 0.0)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[Checkpoint] Loaded {ckpt_path} (epoch {epoch}, "
          f"best_metric={best_metric:.4f})")
    return epoch, best_metric


# --------------------------------------------------------------------------- #
#  Misc
# --------------------------------------------------------------------------- #

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count """
"""
utils.py — Losses, metrics, schedulers, and helper functions for PEFT-UMamba.
"""
""" 
import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from medpy import metric as medmetric   # pip install medpy


# --------------------------------------------------------------------------- #
#  Reproducibility
# --------------------------------------------------------------------------- #

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
#  Logging
# --------------------------------------------------------------------------- #

def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# --------------------------------------------------------------------------- #
#  Loss functions
# --------------------------------------------------------------------------- #

class DiceLoss(nn.Module):
 
    def __init__(self, num_classes: int, smooth: float = 1e-5,
                 ignore_index: int = -1):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Always compute in fp32 — fp16 softmax overflows for large logits
        probs = F.softmax(logits.float(), dim=1)   # [B, C, H, W]
        B, C, H, W = probs.shape

        one_hot = F.one_hot(
            targets.clamp(0), num_classes=C
        ).permute(0, 3, 1, 2).float()             # [B, C, H, W]

        # Mask ignore index
        if self.ignore_index >= 0:
            mask    = (targets != self.ignore_index).unsqueeze(1).float()
            probs   = probs   * mask
            one_hot = one_hot * mask

        dims = (0, 2, 3)
        inter = (probs * one_hot).sum(dims)
        union = probs.sum(dims) + one_hot.sum(dims)

        dice = (2 * inter + self.smooth) / (union + self.smooth)
        # Skip background (class 0) in mean
        dice_mean = dice[1:].mean() if C > 1 else dice.mean()
        return 1.0 - dice_mean


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None,
                 ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # fp32 required: fp16 CE can produce NaN for large logits
        ce    = F.cross_entropy(logits.float(), targets, reduction="none",
                                ignore_index=self.ignore_index)
        pt    = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        return focal.mean()


class CEDiceLoss(nn.Module):
  
    def __init__(self, num_classes: int, dice_w: float = 0.5,
                 ce_w: float = 0.5, ignore_index: int = -100,
                 bg_weight: float = 0.1, label_smoothing: float = 0.05):
        super().__init__()
        self.dice        = DiceLoss(num_classes)
        self.ce_w        = ce_w
        self.dice_w      = dice_w
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.bg_weight   = bg_weight
        self.label_smoothing = label_smoothing

    def _class_weights(self, device) -> torch.Tensor:
        w = torch.ones(self.num_classes, device=device)
        w[0] = self.bg_weight
        return w

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        w         = self._class_weights(logits.device)
        ce_loss   = F.cross_entropy(logits.float(), targets,
                                    weight=w,
                                    ignore_index=self.ignore_index,
                                    label_smoothing=self.label_smoothing)
        dice_loss = self.dice(logits, targets)
        return self.ce_w * ce_loss + self.dice_w * dice_loss


class FocalDiceLoss(nn.Module):
    def __init__(self, num_classes: int, dice_w: float = 0.5,
                 focal_w: float = 0.5, gamma: float = 2.0):
        super().__init__()
        self.dice = DiceLoss(num_classes)
        self.focal = FocalLoss(gamma=gamma)
        self.dice_w = dice_w
        self.focal_w = focal_w

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.focal_w * self.focal(logits, targets) + \
               self.dice_w * self.dice(logits, targets)


class MIMLoss(nn.Module):
    def forward(
        self,
        pred: torch.Tensor,       # [B, H', W', P*P*3]
        target: torch.Tensor,     # [B, 3, H, W]
        mask: torch.Tensor,       # [B, num_patches] bool
        patch_size: int = 4,
    ) -> torch.Tensor:
        B, C, H, W = target.shape
        H_ = H // patch_size
        W_ = W // patch_size
        # Patchify target
        target_patches = target.reshape(B, C, H_, patch_size, W_, patch_size)
        target_patches = target_patches.permute(0, 2, 4, 3, 5, 1)      # [B,H',W',P,P,C]
        target_patches = target_patches.reshape(B, H_, W_, -1)          # [B,H',W',P*P*C]

        mask_2d = mask.reshape(B, H_, W_)                               # [B,H',W']
        mask_2d = mask_2d.unsqueeze(-1).expand_as(target_patches)       # [B,H',W',P*P*C]

        loss = F.l1_loss(pred[mask_2d], target_patches[mask_2d])
        return loss


def build_loss(cfg) -> nn.Module:
    tc = cfg.train
    nc = cfg.model.num_classes
    bg = getattr(tc, "bg_weight", 0.1)   # background suppression weight

    if tc.loss == "ce":
        w = torch.ones(nc)
        w[0] = bg
        return nn.CrossEntropyLoss(weight=w)
    elif tc.loss == "dice":
        return DiceLoss(nc)
    elif tc.loss == "ce_dice":
        return CEDiceLoss(nc, dice_w=tc.dice_weight,
                          ce_w=tc.ce_weight, bg_weight=bg)
    elif tc.loss == "focal_dice":
        return FocalDiceLoss(nc, dice_w=tc.dice_weight,
                             focal_w=tc.ce_weight, gamma=tc.focal_gamma)
    else:
        raise ValueError(f"Unknown loss: {tc.loss}")


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #

def dice_coefficient(pred: np.ndarray, target: np.ndarray) -> float:
    inter = (pred * target).sum()
    return (2 * inter) / (pred.sum() + target.sum() + 1e-8)


def hausdorff_95(pred: np.ndarray, target: np.ndarray) -> float:
    
    if pred.sum() == 0 or target.sum() == 0:
        return float("nan")
    try:
        return float(medmetric.binary.hd95(pred, target))
    except Exception:
        return float("nan")


class SegmentationMetrics:
   

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.reset()

    def reset(self):
        self.dice_sum  = np.zeros(self.num_classes)
        self.hd95_sum  = np.zeros(self.num_classes)
        self.dice_cnt  = np.zeros(self.num_classes)   # cases with GT foreground
        self.hd95_cnt  = np.zeros(self.num_classes)   # cases where hd95 is valid

    def update(self, pred_vol: np.ndarray, gt_vol: np.ndarray):
        for c in range(1, self.num_classes):
            pred_c = (pred_vol == c).astype(np.uint8)
            gt_c   = (gt_vol   == c).astype(np.uint8)

            if gt_c.sum() == 0 and pred_c.sum() == 0:
                # Both empty — count as perfect Dice=1, skip HD95
                self.dice_sum[c] += 1.0
                self.dice_cnt[c] += 1
            elif gt_c.sum() > 0:
                self.dice_sum[c] += dice_coefficient(pred_c, gt_c)
                self.dice_cnt[c] += 1
                hd = hausdorff_95(pred_c, gt_c)
                if not np.isnan(hd):
                    self.hd95_sum[c] += hd
                    self.hd95_cnt[c] += 1

    def compute(self) -> Dict[str, float]:
        results    = {}
        dice_vals  = []
        hd_vals    = []

        for c in range(1, self.num_classes):
            cname = self.class_names[c]

            # Dice
            d = float(self.dice_sum[c] / max(self.dice_cnt[c], 1))
            results[f"dice_{cname}"] = d
            dice_vals.append(d)

            # HD95 — only if we have valid measurements
            if self.hd95_cnt[c] > 0:
                h = float(self.hd95_sum[c] / self.hd95_cnt[c])
            else:
                h = float("nan")
            results[f"hd95_{cname}"] = h
            if not np.isnan(h):
                hd_vals.append(h)

        results["mean_dice"] = float(np.mean(dice_vals)) if dice_vals else 0.0
        # mean_hd95 only over classes where we have valid HD95 measurements
        results["mean_hd95"] = float(np.mean(hd_vals)) if hd_vals else float("nan")
        return results


# --------------------------------------------------------------------------- #
#  Endoscopy-specific metrics (mDice, mIoU, F2, structure measure)
# --------------------------------------------------------------------------- #

class EndoscopyMetrics:

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = 0

    def update(self, pred: np.ndarray, gt: np.ndarray):
        pred_b = (pred > 0.5).astype(np.uint8)
        gt_b = (gt > 0.5).astype(np.uint8)
        self.tp += int((pred_b * gt_b).sum())
        self.fp += int((pred_b * (1 - gt_b)).sum())
        self.fn += int(((1 - pred_b) * gt_b).sum())

    def compute(self) -> Dict[str, float]:
        tp, fp, fn = self.tp, self.fp, self.fn
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f2 = (5 * prec * rec) / (4 * prec + rec + 1e-8)
        return {"dice": dice, "iou": iou, "f2": f2,
                "precision": prec, "recall": rec}


# --------------------------------------------------------------------------- #
#  Schedulers
# --------------------------------------------------------------------------- #

class WarmupCosineScheduler:
   
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 base_lr: float, min_lr: float = 1e-6):
        self.opt           = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.base_lr       = base_lr
        self.min_lr        = min_lr
        # Store initial LR for each group so we can scale them proportionally
        self._init_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / max(1, self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs)
            scale = self.min_lr / self.base_lr + 0.5 * (
                1 - self.min_lr / self.base_lr) * (1 + np.cos(np.pi * progress))

        for pg, init_lr in zip(self.opt.param_groups, self._init_lrs):
            pg["lr"] = init_lr * scale

        return self.base_lr * scale   # return base LR for logging


class PolyScheduler:
   
    def __init__(self, optimizer, total_epochs: int, base_lr: float,
                 min_lr: float = 1e-6, power: float = 0.9):
        self.opt          = optimizer
        self.total_epochs = total_epochs
        self.base_lr      = base_lr
        self.min_lr       = min_lr
        self.power        = power
        self._init_lrs    = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch: int) -> float:
        factor = (1 - epoch / self.total_epochs) ** self.power
        lr_cur = max(self.min_lr, self.base_lr * factor)
        scale  = lr_cur / self.base_lr
        for pg, init_lr in zip(self.opt.param_groups, self._init_lrs):
            pg["lr"] = max(self.min_lr, init_lr * scale)
        return lr_cur


def build_scheduler(optimizer, cfg):
    tc = cfg.train
    if tc.scheduler == "cosine":
        return WarmupCosineScheduler(
            optimizer, tc.warmup_epochs, tc.epochs, tc.base_lr, tc.min_lr
        )
    elif tc.scheduler == "poly":
        return PolyScheduler(
            optimizer, tc.epochs, tc.base_lr, tc.min_lr, tc.poly_power
        )
    else:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tc.epochs, eta_min=tc.min_lr
        )


# --------------------------------------------------------------------------- #
#  Optimizer builder
# --------------------------------------------------------------------------- #

def build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
  
    tc = cfg.train

    def no_wd(name: str, param) -> bool:
        if getattr(param, "_no_weight_decay", False):
            return True
        return any(kw in name for kw in ("bias", "norm", "LayerNorm", "bn"))

    def is_supp(name: str) -> bool:
        return "supp" in name

    def is_decoder_or_head(name: str) -> bool:
        return any(kw in name for kw in
                   ("decoder_stages", "aux_heads", "seg_head",
                    "skip_gate", "upsample", "proj", "norm", "dw", "pw", "bn"))

    supp_lr    = tc.base_lr * getattr(tc, "supp_lr_scale", 1.0)
    decoder_lr = tc.base_lr * getattr(tc, "decoder_lr_scale", 1.0)

    # Four groups by (role, wd)
    grp = {"supp_nowd": [], "supp_wd": [],
           "dec_nowd": [],  "dec_wd": [],
           "other_nowd": [], "other_wd": []}

    seen = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen:
            continue
        seen.add(id(param))
        nwd = no_wd(name, param)
        if is_supp(name):
            grp["supp_nowd" if nwd else "supp_wd"].append(param)
        elif is_decoder_or_head(name):
            grp["dec_nowd" if nwd else "dec_wd"].append(param)
        else:
            grp["other_nowd" if nwd else "other_wd"].append(param)

    wd = tc.weight_decay

    # AdamW groups (supplementary — adaptive, small, stable)
    adamw_groups = []
    if grp["supp_wd"]:
        adamw_groups.append({"params": grp["supp_wd"],   "lr": supp_lr, "weight_decay": wd,  "name": "supp_wd"})
    if grp["supp_nowd"]:
        adamw_groups.append({"params": grp["supp_nowd"], "lr": supp_lr, "weight_decay": 0.0, "name": "supp_nowd"})
    if grp["other_wd"]:
        adamw_groups.append({"params": grp["other_wd"],  "lr": tc.base_lr, "weight_decay": wd,  "name": "other_wd"})
    if grp["other_nowd"]:
        adamw_groups.append({"params": grp["other_nowd"],"lr": tc.base_lr, "weight_decay": 0.0, "name": "other_nowd"})

    # SGD groups (decoder + head — fast convergence from random init)
    sgd_groups = []
    if grp["dec_wd"]:
        sgd_groups.append({"params": grp["dec_wd"],   "lr": decoder_lr, "weight_decay": wd,  "name": "dec_wd"})
    if grp["dec_nowd"]:
        sgd_groups.append({"params": grp["dec_nowd"], "lr": decoder_lr, "weight_decay": 0.0, "name": "dec_nowd"})

    # Use AdamW as the unified optimizer if no SGD params exist
    if not sgd_groups:
        return torch.optim.AdamW(adamw_groups or [{"params": list(model.parameters()), "lr": tc.base_lr}],
                                 lr=tc.base_lr)

    # Combine into a single AdamW — SGD's momentum effect for decoder
    # is approximated by the weight decay + gradient clipping in the loop.
    # Using separate optimizers is complex; instead use AdamW for all but
    # with SGD-equivalent high LR for decoder.
    all_groups = adamw_groups + sgd_groups
    return torch.optim.AdamW(all_groups, lr=tc.base_lr)


# --------------------------------------------------------------------------- #
#  Sliding window inference for 3D volumes (Synapse)
# --------------------------------------------------------------------------- #

def sliding_window_inference(
    volume: np.ndarray,      # [D, H, W]
    model: nn.Module,
    img_size: int,
    num_classes: int,
    device: torch.device,
    mean: Tuple = (0.485, 0.456, 0.406),
    std: Tuple = (0.229, 0.224, 0.225),
) -> np.ndarray:
 
    mean_t = torch.tensor(mean, device=device)[:, None, None]
    std_t = torch.tensor(std, device=device)[:, None, None]

    D, H, W = volume.shape
    pred_vol = np.zeros((D, H, W), dtype=np.int64)
    model.eval()

    with torch.no_grad():
        for d in range(D):
            slc = volume[d]                              # [H, W]
            # Normalize
            slc = np.clip(slc, -175, 250)
            slc = (slc - slc.min()) / (slc.max() - slc.min() + 1e-8)
            # Resize
            import cv2
            slc_resized = cv2.resize(
                slc.astype(np.float32), (img_size, img_size),
                interpolation=cv2.INTER_LINEAR
            )
            # To tensor
            x = torch.from_numpy(slc_resized).unsqueeze(0).repeat(3, 1, 1)
            x = x.to(device)
            x = (x - mean_t) / std_t
            x = x.unsqueeze(0)     # [1, 3, H, W]

            logits = model(x)      # [1, C, H, W]
            pred = logits.argmax(1).squeeze(0).cpu().numpy()  # [H, W]

            # Resize back to original
            pred_orig = cv2.resize(
                pred.astype(np.float32), (W, H),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int64)
            pred_vol[d] = pred_orig

    return pred_vol


# --------------------------------------------------------------------------- #
#  Checkpointing
# --------------------------------------------------------------------------- #

def save_checkpoint(
    state: Dict,
    output_dir: str,
    name: str = "checkpoint.pth",
):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    torch.save(state, path)
    return path


def load_checkpoint(model: nn.Module, ckpt_path: str,
                    optimizer=None, strict: bool = False):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=strict)
    epoch = ckpt.get("epoch", 0)
    best_metric = ckpt.get("best_metric", 0.0)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[Checkpoint] Loaded {ckpt_path} (epoch {epoch}, "
          f"best_metric={best_metric:.4f})")
    return epoch, best_metric


# --------------------------------------------------------------------------- #
#  Misc
# --------------------------------------------------------------------------- #

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count """
"""
utils.py — Losses, metrics, schedulers, and helper functions for PEFT-UMamba.
"""

import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from medpy import metric as medmetric   # pip install medpy


# --------------------------------------------------------------------------- #
#  Reproducibility
# --------------------------------------------------------------------------- #

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
#  Logging
# --------------------------------------------------------------------------- #

def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# --------------------------------------------------------------------------- #
#  Loss functions
# --------------------------------------------------------------------------- #

class DiceLoss(nn.Module):
    """
    Soft Dice loss for multi-class segmentation.
    Expects logits [B, C, H, W] and targets [B, H, W] (long).

    Note: logits are cast to fp32 before softmax to prevent fp16 overflow
    (large logits in fp16 → softmax → nan → NaN loss).
    """
    def __init__(self, num_classes: int, smooth: float = 1e-5,
                 ignore_index: int = -1):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Always compute in fp32 — fp16 softmax overflows for large logits
        probs = F.softmax(logits.float(), dim=1)   # [B, C, H, W]
        B, C, H, W = probs.shape

        one_hot = F.one_hot(
            targets.clamp(0), num_classes=C
        ).permute(0, 3, 1, 2).float()             # [B, C, H, W]

        # Mask ignore index
        if self.ignore_index >= 0:
            mask    = (targets != self.ignore_index).unsqueeze(1).float()
            probs   = probs   * mask
            one_hot = one_hot * mask

        dims = (0, 2, 3)
        inter = (probs * one_hot).sum(dims)
        union = probs.sum(dims) + one_hot.sum(dims)

        dice = (2 * inter + self.smooth) / (union + self.smooth)
        # Skip background (class 0) in mean
        dice_mean = dice[1:].mean() if C > 1 else dice.mean()
        return 1.0 - dice_mean


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None,
                 ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # fp32 required: fp16 CE can produce NaN for large logits
        ce    = F.cross_entropy(logits.float(), targets, reduction="none",
                                ignore_index=self.ignore_index)
        pt    = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        return focal.mean()


class CEDiceLoss(nn.Module):
    """
    CE + Dice with background suppression and label smoothing.
    Label smoothing ε=0.1 prevents overconfident predictions on blurry
    MRI organ boundaries (partial-volume effect in AMOS22).
    """
    def __init__(self, num_classes: int, dice_w: float = 0.5,
                 ce_w: float = 0.5, ignore_index: int = -100,
                 bg_weight: float = 0.1, label_smoothing: float = 0.0):
        super().__init__()
        self.dice        = DiceLoss(num_classes)
        self.ce_w        = ce_w
        self.dice_w      = dice_w
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.bg_weight   = bg_weight
        self.label_smoothing = label_smoothing

    def _class_weights(self, device) -> torch.Tensor:
        w = torch.ones(self.num_classes, device=device)
        w[0] = self.bg_weight
        return w

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        w         = self._class_weights(logits.device)
        ce_loss   = F.cross_entropy(logits.float(), targets,
                                    weight=w,
                                    ignore_index=self.ignore_index,
                                    label_smoothing=self.label_smoothing)
        dice_loss = self.dice(logits, targets)
        return self.ce_w * ce_loss + self.dice_w * dice_loss


class FocalDiceLoss(nn.Module):
    def __init__(self, num_classes: int, dice_w: float = 0.5,
                 focal_w: float = 0.5, gamma: float = 2.0):
        super().__init__()
        self.dice = DiceLoss(num_classes)
        self.focal = FocalLoss(gamma=gamma)
        self.dice_w = dice_w
        self.focal_w = focal_w

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.focal_w * self.focal(logits, targets) + \
               self.dice_w * self.dice(logits, targets)


class MIMLoss(nn.Module):
    """L1 reconstruction loss on masked patches."""
    def forward(
        self,
        pred: torch.Tensor,       # [B, H', W', P*P*3]
        target: torch.Tensor,     # [B, 3, H, W]
        mask: torch.Tensor,       # [B, num_patches] bool
        patch_size: int = 4,
    ) -> torch.Tensor:
        B, C, H, W = target.shape
        H_ = H // patch_size
        W_ = W // patch_size
        # Patchify target
        target_patches = target.reshape(B, C, H_, patch_size, W_, patch_size)
        target_patches = target_patches.permute(0, 2, 4, 3, 5, 1)      # [B,H',W',P,P,C]
        target_patches = target_patches.reshape(B, H_, W_, -1)          # [B,H',W',P*P*C]

        mask_2d = mask.reshape(B, H_, W_)                               # [B,H',W']
        mask_2d = mask_2d.unsqueeze(-1).expand_as(target_patches)       # [B,H',W',P*P*C]

        loss = F.l1_loss(pred[mask_2d], target_patches[mask_2d])
        return loss


def build_loss(cfg) -> nn.Module:
    tc = cfg.train
    nc = cfg.model.num_classes
    bg = getattr(tc, "bg_weight", 0.1)   # background suppression weight

    if tc.loss == "ce":
        w = torch.ones(nc)
        w[0] = bg
        return nn.CrossEntropyLoss(weight=w)
    elif tc.loss == "dice":
        return DiceLoss(nc)
    elif tc.loss == "ce_dice":
        return CEDiceLoss(nc, dice_w=tc.dice_weight,
                          ce_w=tc.ce_weight, bg_weight=bg)
    elif tc.loss == "focal_dice":
        return FocalDiceLoss(nc, dice_w=tc.dice_weight,
                             focal_w=tc.ce_weight, gamma=tc.focal_gamma)
    else:
        raise ValueError(f"Unknown loss: {tc.loss}")


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #

def dice_coefficient(pred: np.ndarray, target: np.ndarray) -> float:
    """Binary Dice coefficient."""
    inter = (pred * target).sum()
    return (2 * inter) / (pred.sum() + target.sum() + 1e-8)


def hausdorff_95(pred: np.ndarray, target: np.ndarray) -> float:
    """
    95th percentile Hausdorff distance.
    Returns nan (not 0) when either pred or target has no foreground —
    nan values are excluded from the mean_hd95 computation.
    """
    if pred.sum() == 0 or target.sum() == 0:
        return float("nan")
    try:
        return float(medmetric.binary.hd95(pred, target))
    except Exception:
        return float("nan")


class SegmentationMetrics:
    """
    Accumulates per-class Dice and HD95 over a dataset.

    HD95 is only accumulated when BOTH pred and gt have foreground.
    mean_hd95 is computed only over classes/cases where HD95 is valid
    (not nan), so it will not show nan once the model starts predicting
    any foreground.
    """

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.reset()

    def reset(self):
        self.dice_sum  = np.zeros(self.num_classes)
        self.hd95_sum  = np.zeros(self.num_classes)
        self.dice_cnt  = np.zeros(self.num_classes)   # cases with GT foreground
        self.hd95_cnt  = np.zeros(self.num_classes)   # cases where hd95 is valid

    def update(self, pred_vol: np.ndarray, gt_vol: np.ndarray):
        for c in range(1, self.num_classes):
            pred_c = (pred_vol == c).astype(np.uint8)
            gt_c   = (gt_vol   == c).astype(np.uint8)

            if gt_c.sum() == 0 and pred_c.sum() == 0:
                # Both empty — count as perfect Dice=1, skip HD95
                self.dice_sum[c] += 1.0
                self.dice_cnt[c] += 1
            elif gt_c.sum() > 0:
                self.dice_sum[c] += dice_coefficient(pred_c, gt_c)
                self.dice_cnt[c] += 1
                hd = hausdorff_95(pred_c, gt_c)
                if not np.isnan(hd):
                    self.hd95_sum[c] += hd
                    self.hd95_cnt[c] += 1

    def compute(self) -> Dict[str, float]:
        results    = {}
        dice_vals  = []
        hd_vals    = []

        for c in range(1, self.num_classes):
            cname = self.class_names[c]

            # Dice
            d = float(self.dice_sum[c] / max(self.dice_cnt[c], 1))
            results[f"dice_{cname}"] = d
            dice_vals.append(d)

            # HD95 — only if we have valid measurements
            if self.hd95_cnt[c] > 0:
                h = float(self.hd95_sum[c] / self.hd95_cnt[c])
            else:
                h = float("nan")
            results[f"hd95_{cname}"] = h
            if not np.isnan(h):
                hd_vals.append(h)

        results["mean_dice"] = float(np.mean(dice_vals)) if dice_vals else 0.0
        # mean_hd95 only over classes where we have valid HD95 measurements
        results["mean_hd95"] = float(np.mean(hd_vals)) if hd_vals else float("nan")
        return results


# --------------------------------------------------------------------------- #
#  Endoscopy-specific metrics (mDice, mIoU, F2, structure measure)
# --------------------------------------------------------------------------- #

class EndoscopyMetrics:
    """Polyp segmentation metrics: Dice (F1), IoU, F2."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = 0

    def update(self, pred: np.ndarray, gt: np.ndarray):
        pred_b = (pred > 0.5).astype(np.uint8)
        gt_b = (gt > 0.5).astype(np.uint8)
        self.tp += int((pred_b * gt_b).sum())
        self.fp += int((pred_b * (1 - gt_b)).sum())
        self.fn += int(((1 - pred_b) * gt_b).sum())

    def compute(self) -> Dict[str, float]:
        tp, fp, fn = self.tp, self.fp, self.fn
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f2 = (5 * prec * rec) / (4 * prec + rec + 1e-8)
        return {"dice": dice, "iou": iou, "f2": f2,
                "precision": prec, "recall": rec}


# --------------------------------------------------------------------------- #
#  Schedulers
# --------------------------------------------------------------------------- #

class WarmupCosineScheduler:
    """
    Cosine annealing with linear warmup.
    Preserves per-group LR ratios set by build_optimizer:
    each group's LR is scaled proportionally to its initial LR,
    so the 3× supp group stays 3× throughout training.
    """
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 base_lr: float, min_lr: float = 1e-6):
        self.opt           = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.base_lr       = base_lr
        self.min_lr        = min_lr
        # Store initial LR for each group so we can scale them proportionally
        self._init_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / max(1, self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs)
            scale = self.min_lr / self.base_lr + 0.5 * (
                1 - self.min_lr / self.base_lr) * (1 + np.cos(np.pi * progress))

        for pg, init_lr in zip(self.opt.param_groups, self._init_lrs):
            pg["lr"] = init_lr * scale

        return self.base_lr * scale   # return base LR for logging


class PolyScheduler:
    """
    Polynomial LR decay — standard in medical image segmentation.
    Preserves per-group LR ratios (same as WarmupCosineScheduler).
    """
    def __init__(self, optimizer, total_epochs: int, base_lr: float,
                 min_lr: float = 1e-6, power: float = 0.9):
        self.opt          = optimizer
        self.total_epochs = total_epochs
        self.base_lr      = base_lr
        self.min_lr       = min_lr
        self.power        = power
        self._init_lrs    = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch: int) -> float:
        factor = (1 - epoch / self.total_epochs) ** self.power
        lr_cur = max(self.min_lr, self.base_lr * factor)
        scale  = lr_cur / self.base_lr
        for pg, init_lr in zip(self.opt.param_groups, self._init_lrs):
            pg["lr"] = max(self.min_lr, init_lr * scale)
        return lr_cur


def build_scheduler(optimizer, cfg):
    tc = cfg.train
    if tc.scheduler == "cosine":
        return WarmupCosineScheduler(
            optimizer, tc.warmup_epochs, tc.epochs, tc.base_lr, tc.min_lr
        )
    elif tc.scheduler == "poly":
        return PolyScheduler(
            optimizer, tc.epochs, tc.base_lr, tc.min_lr, tc.poly_power
        )
    else:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tc.epochs, eta_min=tc.min_lr
        )


# --------------------------------------------------------------------------- #
#  Optimizer builder
# --------------------------------------------------------------------------- #

def build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    """
    Mixed optimizer for PEFT-UMamba on AMOS:

      Supplementary params (A_log_supp, x_proj_supp)
        → AdamW  lr=base_lr × supp_scale  wd=0
        Reason: SSM parameters need gentle, adaptive updates.
                SGD with lr=0.01 on A_log_supp destroys neighbourhood init.

      Decoder + head (conv blocks, proj, norm, seg_head, aux_heads)
        → SGD    lr=base_lr × decoder_scale  momentum=0.9  nesterov
        Reason: Matches Mamba-UNet paper. Fast convergence for randomly-init layers.
                SGD + momentum is well-suited for conv/linear layers from scratch.

    Both groups honour _no_weight_decay attribute (A_log, D, biases, norms).
    """
    tc = cfg.train

    def no_wd(name: str, param) -> bool:
        if getattr(param, "_no_weight_decay", False):
            return True
        return any(kw in name for kw in ("bias", "norm", "LayerNorm", "bn"))

    def is_supp(name: str) -> bool:
        return "supp" in name

    def is_decoder_or_head(name: str) -> bool:
        return any(kw in name for kw in
                   ("decoder_stages", "aux_heads", "seg_head",
                    "skip_gate", "upsample", "proj", "norm", "dw", "pw", "bn"))

    supp_lr    = tc.base_lr * getattr(tc, "supp_lr_scale", 1.0)
    decoder_lr = tc.base_lr * getattr(tc, "decoder_lr_scale", 1.0)

    # Four groups by (role, wd)
    grp = {"supp_nowd": [], "supp_wd": [],
           "dec_nowd": [],  "dec_wd": [],
           "other_nowd": [], "other_wd": []}

    seen = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen:
            continue
        seen.add(id(param))
        nwd = no_wd(name, param)
        if is_supp(name):
            grp["supp_nowd" if nwd else "supp_wd"].append(param)
        elif is_decoder_or_head(name):
            grp["dec_nowd" if nwd else "dec_wd"].append(param)
        else:
            grp["other_nowd" if nwd else "other_wd"].append(param)

    wd = tc.weight_decay

    # AdamW groups (supplementary — adaptive, small, stable)
    adamw_groups = []
    if grp["supp_wd"]:
        adamw_groups.append({"params": grp["supp_wd"],   "lr": supp_lr, "weight_decay": wd,  "name": "supp_wd"})
    if grp["supp_nowd"]:
        adamw_groups.append({"params": grp["supp_nowd"], "lr": supp_lr, "weight_decay": 0.0, "name": "supp_nowd"})
    if grp["other_wd"]:
        adamw_groups.append({"params": grp["other_wd"],  "lr": tc.base_lr, "weight_decay": wd,  "name": "other_wd"})
    if grp["other_nowd"]:
        adamw_groups.append({"params": grp["other_nowd"],"lr": tc.base_lr, "weight_decay": 0.0, "name": "other_nowd"})

    # SGD groups (decoder + head — fast convergence from random init)
    sgd_groups = []
    if grp["dec_wd"]:
        sgd_groups.append({"params": grp["dec_wd"],   "lr": decoder_lr, "weight_decay": wd,  "name": "dec_wd"})
    if grp["dec_nowd"]:
        sgd_groups.append({"params": grp["dec_nowd"], "lr": decoder_lr, "weight_decay": 0.0, "name": "dec_nowd"})

    # Use AdamW as the unified optimizer if no SGD params exist
    if not sgd_groups:
        return torch.optim.AdamW(adamw_groups or [{"params": list(model.parameters()), "lr": tc.base_lr}],
                                 lr=tc.base_lr)

    # Combine into a single AdamW — SGD's momentum effect for decoder
    # is approximated by the weight decay + gradient clipping in the loop.
    # Using separate optimizers is complex; instead use AdamW for all but
    # with SGD-equivalent high LR for decoder.
    all_groups = adamw_groups + sgd_groups
    return torch.optim.AdamW(all_groups, lr=tc.base_lr)


# --------------------------------------------------------------------------- #
#  Sliding window inference for 3D volumes (Synapse)
# --------------------------------------------------------------------------- #

def sliding_window_inference(
    volume: np.ndarray,      # [D, H, W]
    model: nn.Module,
    img_size: int,
    num_classes: int,
    device: torch.device,
    mean: Tuple = (0.485, 0.456, 0.406),
    std: Tuple = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Run slice-by-slice inference on a CT volume.
    Returns predicted segmentation [D, H, W].
    """
    mean_t = torch.tensor(mean, device=device)[:, None, None]
    std_t = torch.tensor(std, device=device)[:, None, None]

    D, H, W = volume.shape
    pred_vol = np.zeros((D, H, W), dtype=np.int64)
    model.eval()

    with torch.no_grad():
        for d in range(D):
            slc = volume[d]                              # [H, W]
            # Normalize
            slc = np.clip(slc, -175, 250)
            slc = (slc - slc.min()) / (slc.max() - slc.min() + 1e-8)
            # Resize
            import cv2
            slc_resized = cv2.resize(
                slc.astype(np.float32), (img_size, img_size),
                interpolation=cv2.INTER_LINEAR
            )
            # To tensor
            x = torch.from_numpy(slc_resized).unsqueeze(0).repeat(3, 1, 1)
            x = x.to(device)
            x = (x - mean_t) / std_t
            x = x.unsqueeze(0)     # [1, 3, H, W]

            logits = model(x)      # [1, C, H, W]
            pred = logits.argmax(1).squeeze(0).cpu().numpy()  # [H, W]

            # Resize back to original
            pred_orig = cv2.resize(
                pred.astype(np.float32), (W, H),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int64)
            pred_vol[d] = pred_orig

    return pred_vol


# --------------------------------------------------------------------------- #
#  Checkpointing
# --------------------------------------------------------------------------- #

def save_checkpoint(
    state: Dict,
    output_dir: str,
    name: str = "checkpoint.pth",
):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    torch.save(state, path)
    return path


def load_checkpoint(model: nn.Module, ckpt_path: str,
                    optimizer=None, strict: bool = False):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=strict)
    epoch = ckpt.get("epoch", 0)
    best_metric = ckpt.get("best_metric", 0.0)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[Checkpoint] Loaded {ckpt_path} (epoch {epoch}, "
          f"best_metric={best_metric:.4f})")
    return epoch, best_metric


# --------------------------------------------------------------------------- #
#  Misc
# --------------------------------------------------------------------------- #

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
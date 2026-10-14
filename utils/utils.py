"""
utils/utils.py — loss, optimiser, schedule, metrics and bookkeeping (§4.2–4.3).
"""

import logging
import os
import random
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================== #
#  Reproducibility, logging, bookkeeping
# =========================================================================== #

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(name: str, log_file: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)):
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.propagate = False
    return logger


class AverageMeter:
    def __init__(self):
        self.sum, self.n = 0.0, 0

    def update(self, v: float, k: int = 1):
        self.sum += v * k
        self.n += k

    @property
    def avg(self):
        return self.sum / max(self.n, 1)


def save_checkpoint(state: dict, out_dir: str, name: str):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(state, os.path.join(out_dir, name))


def load_checkpoint(model, path: str, optimizer=None, scheduler=None) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt.get("epoch", 0))


def count_params(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters()
               if p.requires_grad or not trainable_only)


@torch.no_grad()
def measure_inference_ms(model, img_size: int, device, batch: int = 1,
                         n_warmup: int = 20, n_runs: int = 100) -> float:
    model.eval()
    x = torch.randn(batch, 3, img_size, img_size, device=device)
    for _ in range(n_warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_runs * 1000.0


# =========================================================================== #
#  Loss: CE + Dice with deep supervision
# =========================================================================== #

class DiceCELoss(nn.Module):
    """Cross-entropy + soft Dice (batch Dice over foreground classes)."""

    def __init__(self, num_classes: int, ce_weight: float = 1.0,
                 dice_weight: float = 1.0, smooth: float = 1e-5):
        super().__init__()
        self.C, self.wc, self.wd, self.smooth = num_classes, ce_weight, dice_weight, smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        ce = F.cross_entropy(logits, target)
        probs = logits.softmax(1)
        onehot = F.one_hot(target, self.C).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probs * onehot).sum(dims)
        denom = probs.sum(dims) + onehot.sum(dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return self.wc * ce + self.wd * (1.0 - dice[1:].mean())


def deep_supervision_loss(criterion, logits, aux: Sequence[torch.Tensor],
                          target, weights: Sequence[float], enabled: bool):
    """L = L(main) + Σ_i w_i L(aux_i), aux ordered deepest → shallowest."""
    loss = criterion(logits, target)
    if enabled:
        for w, a in zip(weights, aux):
            loss = loss + w * criterion(a, target)
    return loss


# =========================================================================== #
#  Optimiser and schedule
# =========================================================================== #

_NO_DECAY_KEYS = ("A_logs", "Ds", "dt_projs_bias")


def build_optimizer(model: nn.Module, lr: float, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or any(k in name for k in _NO_DECAY_KEYS):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def build_scheduler(optimizer, total_iters: int, warmup_iters: int,
                    power: float = 0.9):
    """Linear warm-up, then polynomial decay; stepped once per iteration."""
    def fn(it):
        if it < warmup_iters:
            return (it + 1) / max(warmup_iters, 1)
        t = (it - warmup_iters) / max(total_iters - warmup_iters, 1)
        return max(1.0 - t, 0.0) ** power
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


# =========================================================================== #
#  Metrics
# =========================================================================== #

def dice_per_class(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> np.ndarray:
    """Dice for classes 1..C-1; NaN when the class is absent in both."""
    out = np.full(num_classes - 1, np.nan, dtype=np.float64)
    for c in range(1, num_classes):
        p, g = pred == c, gt == c
        ps, gs = p.sum(), g.sum()
        if ps == 0 and gs == 0:
            continue
        out[c - 1] = 2.0 * np.logical_and(p, g).sum() / (ps + gs)
    return out


def _surface(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage
    return np.logical_xor(mask, ndimage.binary_erosion(mask))


def hd95_per_class(pred: np.ndarray, gt: np.ndarray, num_classes: int,
                   spacing: Sequence[float]) -> np.ndarray:
    """95th-percentile Hausdorff distance (in mm when spacing is in mm);
    NaN when the class is empty in prediction or ground truth."""
    from scipy import ndimage
    out = np.full(num_classes - 1, np.nan, dtype=np.float64)
    for c in range(1, num_classes):
        p, g = pred == c, gt == c
        if not p.any() or not g.any():
            continue
        sp, sg = _surface(p), _surface(g)
        dt_g = ndimage.distance_transform_edt(~sg, sampling=spacing)
        dt_p = ndimage.distance_transform_edt(~sp, sampling=spacing)
        d = np.concatenate([dt_g[sp], dt_p[sg]])
        out[c - 1] = np.percentile(d, 95)
    return out


def instance_f1(pred: np.ndarray, gt: np.ndarray, interior_class: int = 1,
                iou_thr: float = 0.5) -> float:
    """Instance F1 at IoU>0.5. Instances = connected components of the
    cell-interior class in prediction and ground truth."""
    from scipy import ndimage
    pl, npred = ndimage.label(pred == interior_class)
    gl, ngt = ndimage.label(gt == interior_class)
    if npred == 0 and ngt == 0:
        return 1.0
    if npred == 0 or ngt == 0:
        return 0.0
    counts = np.bincount((gl.astype(np.int64) * (npred + 1) + pl).ravel(),
                         minlength=(ngt + 1) * (npred + 1)).reshape(ngt + 1, npred + 1)
    inter = counts[1:, 1:]
    area_g = counts[1:, :].sum(1)
    area_p = counts[:, 1:].sum(0)
    iou = inter / (area_g[:, None] + area_p[None, :] - inter + 1e-9)
    tp = int((iou > iou_thr).sum())          # IoU > 0.5 matches are unique
    return 2.0 * tp / (npred + ngt)


class SegMetrics:
    """Per-case accumulation; mean ± std are computed ACROSS CASES."""

    def __init__(self, num_classes: int, class_names: List[str],
                 compute_hd95: bool = False, compute_instance_f1: bool = False):
        self.C, self.names = num_classes, class_names
        self.hd, self.inst = compute_hd95, compute_instance_f1
        self.dice, self.hd95, self.f1 = [], [], []

    def add_case(self, pred: np.ndarray, gt: np.ndarray,
                 spacing: Optional[Sequence[float]] = None):
        self.dice.append(dice_per_class(pred, gt, self.C))
        if self.hd:
            sp = spacing if spacing is not None else (1.0,) * gt.ndim
            self.hd95.append(hd95_per_class(pred, gt, self.C, sp))
        if self.inst:
            self.f1.append(instance_f1(pred, gt))

    def compute(self) -> Dict[str, float]:
        d = np.stack(self.dice)                       # [cases, C-1]
        case_mean = np.nanmean(d, axis=1)
        res = {"dice": float(np.nanmean(case_mean)),
               "dice_std": float(np.nanstd(case_mean)),
               "n_cases": int(len(case_mean))}
        for i, n in enumerate(self.names[1:]):
            res[f"dice_{n}"] = float(np.nanmean(d[:, i])) if np.any(~np.isnan(d[:, i])) else float("nan")
        if self.hd95:
            h = np.stack(self.hd95)
            hm = np.nanmean(h, axis=1)
            res["hd95"] = float(np.nanmean(hm))
            for i, n in enumerate(self.names[1:]):
                res[f"hd95_{n}"] = float(np.nanmean(h[:, i])) if np.any(~np.isnan(h[:, i])) else float("nan")
        if self.f1:
            res["instance_f1"] = float(np.mean(self.f1))
            res["instance_f1_std"] = float(np.std(self.f1))
        return res

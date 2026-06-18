import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from configs.config import config_from_dataset_name
from data.dataset import (
    MIMDataset, build_loaders_from_config,
    get_dataset_info, resolve_dataset_key,
    sliding_window_predict, sliding_window_predict_3d,
)
from models.model import PEFTUMamba, build_model
from utils.utils import (
    AverageMeter, EndoscopyMetrics, MIMLoss, SegmentationMetrics,
    WarmupCosineScheduler, build_loss, build_optimizer, build_scheduler,
    get_logger, load_checkpoint, save_checkpoint, set_seed,
)


# =========================================================================== #
#  Argument parsing — all server paths set as defaults
# =========================================================================== #

def parse_args():
    p = argparse.ArgumentParser(
        description="PEFT-UMamba Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    p.add_argument("--dataset",    type=str, default="dataset702",
                   help="dataset701|dataset702|dataset704|kvasir "
                        "(aliases: abdomenct, amos, endovis17, polyp, synapse)")
    p.add_argument("--data_root",  type=str,
                   default="/workdir1.8t/fei27/CGT/peft_umamba/"
                           "peft_umamba_Amos/data/Dataset702_AbdomenMR")
    p.add_argument("--img_size",   type=int, default=None,
                   help="Override img_size (auto: 224 for MRI/CT, 352 for endoscopy)")
    p.add_argument("--num_classes",type=int, default=None,
                   help="Override num_classes (auto-detected from dataset)")

    # ── Stage ─────────────────────────────────────────────────────────────────
    p.add_argument("--stage",      type=str, default="finetune",
                   choices=["mim_adapt", "finetune"])

    # ── Model ─────────────────────────────────────────────────────────────────
    p.add_argument("--pretrained", type=str,
                   default="/workdir1.8t/fei27/CGT/peft_umamba/"
                           "peft_umamba_Amos/data/vmamba/"
                           "vssm_tiny_0230_ckpt_epoch_262.pth",
                   help="VMamba-Tiny ImageNet checkpoint (.pth)")
    p.add_argument("--resume",     type=str,
                   default="/workdir1.8t/fei27/CGT/peft_umamba/"
                           "peft_umamba_Amos/outputs/"
                           "peft_umamba_dataset702_mr/best_model.pth",
                   help="Resume from checkpoint (set to '' to train from scratch)")
    p.add_argument("--d_state",      type=int, default=16,
                   help="Base SSM state dimension K")
    p.add_argument("--d_state_supp", type=int, default=4,
                   help="Supplementary state dimension K′")
    p.add_argument("--no_supp_scan", action="store_true",
                   help="Disable Supplementary Scan (ablation baseline)")
    p.add_argument("--no_freeze",    action="store_true",
                   help="Disable encoder freezing — full fine-tune mode (ablation)")
    p.add_argument("--supp_init",    type=str, default="neighbourhood",
                   choices=["neighbourhood", "zero", "random_normal",
                            "xavier", "copy_frozen"],
                   help="A_log_supp initialisation strategy (NI ablation)")
    p.add_argument("--use_sdlora",   action="store_true",
                   help="Use Scale-Decoupled LoRA instead of Supplementary Scan")
    p.add_argument("--use_lora",     action="store_true",
                   help="Enable LoRA adaptation (ablation baseline)")
    p.add_argument("--lora_rank",    type=int, default=4,
                   help="LoRA rank r (used with --use_lora or --use_sdlora)")
    p.add_argument("--no_skip_gate",  action="store_true",
                   help="Disable Skip Attention Gates (ablation D)")
    p.add_argument("--aux_weight",    type=float, default=0.2,
                   help="Deep supervision aux loss weight (0.0 disables, ablation E)")
    p.add_argument("--enable_isam",   action="store_true",
                   help="Enable ISAM 3D inter-slice aggregation training")
    p.add_argument("--isam_lr_scale",  type=float, default=2.0,
                   help="LR multiplier for ISAM params (default 2.0)")
    p.add_argument("--unfreeze_epoch", type=int, default=None,
                   help="Epoch to unfreeze encoder stages 2+3 (default: never)")

    # ── Training hyperparameters ──────────────────────────────────────────────
    p.add_argument("--epochs",        type=int,   default=None)
    p.add_argument("--batch_size",    type=int,   default=None)
    p.add_argument("--lr",            type=float, default=None,
                   help="Base learning rate")
    p.add_argument("--weight_decay",  type=float, default=None)
    p.add_argument("--gradient_clip", type=float, default=None)
    p.add_argument("--loss",          type=str,   default=None,
                   choices=["ce", "dice", "ce_dice", "focal_dice"])
    p.add_argument("--scheduler",     type=str,   default="poly",
                   help="LR scheduler: poly (default) or cosine")
    p.add_argument("--no_amp",        action="store_true",
                   help="Disable AMP (train in full fp32)")
    p.add_argument("--supp_lr_scale", type=float, default=None,
                   help="LR multiplier for supplementary params (default 3.0)")
    p.add_argument("--num_workers",   type=int,   default=8,
                   help="DataLoader workers (increase for faster IO)")

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument("--output_dir",    type=str,
                   default="/workdir1.8t/fei27/CGT/peft_umamba/"
                           "peft_umamba_Amos/outputs")
    p.add_argument("--exp_name",      type=str, default=None,
                   help="Experiment sub-folder name (auto from dataset if None)")
    p.add_argument("--log_interval",  type=int, default=10,
                   help="Print training loss every N steps")
    p.add_argument("--save_interval", type=int, default=5,
                   help="Save periodic checkpoint every N epochs (default 5 for crash safety)")

    # ── MIM adaptation ────────────────────────────────────────────────────────
    p.add_argument("--mask_ratio",  type=float, default=0.60)
    p.add_argument("--mim_epochs",  type=int,   default=50)

    # ── Misc ──────────────────────────────────────────────────────────────────
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")

    return p.parse_args()


def apply_overrides(args, cfg):
    """CLI flags override config defaults. Only override if explicitly set."""
    cfg.data.data_root   = args.data_root
    cfg.data.num_workers = args.num_workers

    if args.img_size    is not None:
        cfg.data.img_size    = args.img_size
        cfg.model.img_size   = args.img_size
    if args.num_classes is not None:
        cfg.data.num_classes  = args.num_classes
        cfg.model.num_classes = args.num_classes
    if args.epochs      is not None:
        cfg.train.epochs      = args.epochs
    if args.batch_size  is not None:
        cfg.train.batch_size  = args.batch_size
    if args.lr          is not None:
        cfg.train.base_lr     = args.lr
    if args.weight_decay is not None:
        cfg.train.weight_decay = args.weight_decay
    if args.gradient_clip is not None:
        cfg.train.gradient_clip = args.gradient_clip
    if args.loss        is not None:
        cfg.train.loss        = args.loss
    if args.exp_name    is not None:
        cfg.train.experiment_name = args.exp_name
    if args.save_interval is not None:
        cfg.train.save_interval = args.save_interval
    if args.supp_lr_scale is not None:
        cfg.train.supp_lr_scale = args.supp_lr_scale

    cfg.train.stage          = args.stage
    cfg.train.scheduler      = args.scheduler
    cfg.train.use_amp        = not args.no_amp
    cfg.train.output_dir     = args.output_dir
    cfg.train.seed           = args.seed
    cfg.train.log_interval   = args.log_interval
    # Pass resume path — empty string means start from scratch
    cfg.train.resume         = args.resume if args.resume else None

    cfg.model.pretrained_path             = args.pretrained
    cfg.model.ssm.d_state                 = args.d_state
    cfg.model.peft.supp_state_dim         = args.d_state_supp
    cfg.model.peft.use_supplementary_scan = not args.no_supp_scan
    cfg.model.freeze_encoder              = not args.no_freeze
    cfg.model.peft.supp_init              = args.supp_init
    cfg.model.peft.use_sdlora             = getattr(args, "use_sdlora", False)
    cfg.model.peft.use_lora               = getattr(args, "use_lora", False)
    cfg.model.peft.lora_rank              = getattr(args, "lora_rank", 4)
    cfg.train.unfreeze_epoch              = getattr(args, "unfreeze_epoch", None)
    cfg.train.enable_isam                 = getattr(args, "enable_isam", False)
    cfg.train.isam_lr_scale               = getattr(args, "isam_lr_scale", 2.0)
    cfg.model.peft.no_skip_gate           = getattr(args, "no_skip_gate", False)
    cfg.train.aux_weight                  = getattr(args, "aux_weight", 0.2)

    cfg.mim.mask_ratio = args.mask_ratio
    cfg.mim.epochs     = args.mim_epochs
    return cfg


# =========================================================================== #
#  Logging helpers
# =========================================================================== #

def log_param_groups(optimizer, logger):
    logger.info("Optimizer param groups:")
    for i, pg in enumerate(optimizer.param_groups):
        n    = sum(p.numel() for p in pg["params"])
        name = pg.get("name", f"group_{i}")
        logger.info(f"  [{name:15s}]  params={n/1e6:.3f}M  "
                    f"lr={pg['lr']:.2e}  wd={pg.get('weight_decay', 0):.0e}")
    total = sum(p.numel() for p in optimizer.param_groups[0]["params"]
                if True) if optimizer.param_groups else 0
    logger.info(f"  Total trainable: "
                f"{sum(sum(p.numel() for p in pg['params']) for pg in optimizer.param_groups)/1e6:.3f}M")


# =========================================================================== #
#  Evaluation
# =========================================================================== #

def _is_volume_dataset(key: str) -> bool:
    """Kvasir and NeurIPS Cell use 2-D evaluation; NIfTI datasets use volumetric."""
    return key not in ("kvasir", "dataset703")


@torch.no_grad()
def evaluate(model, val_loader, cfg, device, logger,
             verbose: bool = True) -> Dict:
    """
    verbose=True  : full per-class breakdown
    verbose=False : summary line only + top-5 worst classes (always shown
                    so you can track small-organ progress even on quiet epochs)
    """
    key = resolve_dataset_key(cfg.data.dataset)
    num_classes, modality, class_names = get_dataset_info(key)
    model.eval()

    if _is_volume_dataset(key):
        metrics = SegmentationMetrics(num_classes, class_names=class_names)
        for batch_idx, batch in enumerate(val_loader):
            vol  = batch["image"]
            gt   = batch["label"]

            if isinstance(vol, torch.Tensor):
                vol = vol.squeeze(0).numpy()
            else:
                vol = np.array(vol).squeeze(0)
            if isinstance(gt, torch.Tensor):
                gt = gt.squeeze(0).numpy().astype(np.int64)
            else:
                gt = np.array(gt).squeeze(0).astype(np.int64)

            assert vol.ndim == 3, \
                f"[Eval] Expected [D,H,W], got {vol.shape}"

            pred = (sliding_window_predict_3d
                    if getattr(model, 'use_3d', False)
                    else sliding_window_predict)(
                vol, model, cfg.data.img_size,
                num_classes, device, modality)
            metrics.update(pred, gt)

        results = metrics.compute()
        mhd = results["mean_hd95"]
        hd_str = f"{mhd:.2f}" if not np.isnan(mhd) else "n/a (no pred yet)"
        logger.info(f"[Val] mean_dice={results['mean_dice']:.4f}  "
                    f"mean_hd95={hd_str}")

        if verbose:
            # Full breakdown
            for cn in class_names[1:]:
                d = results.get(f"dice_{cn}", 0.0)
                h = results.get(f"hd95_{cn}", float("nan"))
                h_str = f"{h:.2f}" if not np.isnan(h) else " n/a"
                logger.info(f"       {cn:<22s}  dice={d:.4f}  hd95={h_str}")
        else:
            # Always show 5 worst-performing foreground classes
            fg_classes = class_names[1:]
            fg_dices   = [results.get(f"dice_{cn}", 0.0) for cn in fg_classes]
            worst5_idx = np.argsort(fg_dices)[:5]
            logger.info("       [5 worst classes this epoch]")
            for idx in worst5_idx:
                cn = fg_classes[idx]
                d  = fg_dices[idx]
                h  = results.get(f"hd95_{cn}", float("nan"))
                h_str = f"{h:.2f}" if not np.isnan(h) else " n/a"
                logger.info(f"       {cn:<22s}  dice={d:.4f}  hd95={h_str}")

    else:
        metrics = EndoscopyMetrics()
        for batch in val_loader:
            imgs = batch["image"].to(device)
            gt   = batch["label"].cpu().numpy()
            pred = model(imgs).argmax(1).cpu().numpy()
            for b in range(pred.shape[0]):
                metrics.update(pred[b], gt[b])
        results = metrics.compute()
        logger.info(f"[Val] dice={results['dice']:.4f}  "
                    f"iou={results['iou']:.4f}  "
                    f"f2={results['f2']:.4f}")

    return results


def primary_metric(results: Dict, key: str) -> float:
    return results.get("mean_dice" if _is_volume_dataset(key) else "dice", 0.0)


# =========================================================================== #
#  Stage 2 — MIM domain adaptation
# =========================================================================== #

def run_mim_adaptation(model: PEFTUMamba, cfg, device, logger):
    mc = cfg.mim
    tc = cfg.train
    dc = cfg.data

    logger.info("=" * 60)
    logger.info("Stage 2: MIM Domain Adaptation")
    logger.info(f"  dataset={dc.dataset}  mask_ratio={mc.mask_ratio}  "
                f"epochs={mc.epochs}")
    logger.info("=" * 60)

    model.add_mim_head(patch_size=4, in_chans=3)
    model = model.to(device)
    if cfg.model.peft.use_lora:
        model.enable_lora(cfg.model.peft.lora_rank, cfg.model.peft.lora_alpha)

    root     = Path(dc.data_root)
    img_dirs = [str(d) for sub in
                ("imagesTr", "imagesTs", "images",
                 "Kvasir-SEG/images", "kvasir/Kvasir-SEG/images")
                if (d := root / sub).exists()]
    if not img_dirs:
        img_dirs = [str(root)]

    mim_ds = MIMDataset(
        img_dirs, img_size=dc.img_size, patch_size=4,
        mask_ratio=mc.mask_ratio, modality=dc.modality)
    loader = DataLoader(
        mim_ds, batch_size=tc.batch_size, shuffle=True,
        num_workers=dc.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(dc.num_workers > 0),
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=mc.base_lr, weight_decay=mc.weight_decay)
    scheduler = WarmupCosineScheduler(
        optimizer, mc.warmup_epochs, mc.epochs, mc.base_lr, mc.min_lr)
    criterion = MIMLoss()
    scaler    = GradScaler(enabled=tc.use_amp)
    out_dir   = os.path.join(tc.output_dir, tc.experiment_name, "mim")
    os.makedirs(out_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(mc.epochs):
        model.train()
        meter = AverageMeter()
        lr    = scheduler.step(epoch)

        for step, batch in enumerate(loader):
            imgs  = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device,  non_blocking=True)

            with autocast(enabled=tc.use_amp):
                pred = model(imgs, mim_mask=masks)
                loss = criterion(pred, imgs, masks, patch_size=4)

            if not loss.isfinite():
                logger.warning(f"[MIM] NaN/Inf at epoch {epoch+1} step {step}")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), tc.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            meter.update(loss.item(), imgs.size(0))

            if step % tc.log_interval == 0:
                logger.info(f"[MIM] Ep{epoch+1}/{mc.epochs} "
                            f"Step{step}/{len(loader)} "
                            f"Loss={meter.avg:.4f} LR={lr:.2e}")

        logger.info(f"[MIM] Epoch {epoch+1} avg loss={meter.avg:.4f}")
        if meter.avg < best_loss:
            best_loss = meter.avg
            save_checkpoint({"epoch": epoch, "model": model.state_dict(),
                             "best_loss": best_loss}, out_dir, "best_mim.pth")

    save_checkpoint({"epoch": mc.epochs, "model": model.state_dict()},
                    out_dir, "final_mim.pth")
    logger.info(f"[MIM] Done.  Best loss: {best_loss:.4f}")
    return model


# =========================================================================== #
#  Stage 3 — PEFT fine-tuning  (main training stage)
# =========================================================================== #

def run_finetune(model: PEFTUMamba, cfg, device, logger):
    tc  = cfg.train
    key = resolve_dataset_key(cfg.data.dataset)

    logger.info("=" * 60)
    logger.info(f"Stage 3: PEFT Fine-Tuning  [{cfg.data.dataset}]")
    logger.info(f"  trainable : {model.trainable_param_count()/1e6:.3f}M  "
                f"/ total : {model.total_param_count()/1e6:.1f}M")
    logger.info("=" * 60)

    model = model.to(device)

    train_loader, val_loader = build_loaders_from_config(cfg)
    criterion = build_loss(cfg)
    optimizer = build_optimizer(model, cfg)
    log_param_groups(optimizer, logger)

    scheduler = build_scheduler(optimizer, cfg)
    scaler    = GradScaler(enabled=tc.use_amp)
    out_dir   = os.path.join(tc.output_dir, tc.experiment_name)
    os.makedirs(out_dir, exist_ok=True)

    start_epoch = 0
    best_metric = 0.0
    nan_total   = 0

    if getattr(tc, "resume", None):
        start_epoch, best_metric = load_checkpoint(model, tc.resume, optimizer)

    # ── Enable ISAM 3D training ───────────────────────────────────────────────
    # The 2D backbone is now fully converged (best DSC=0.6772 at ep192).
    # Activate ISAM so inter-slice aggregation is trained for remaining epochs.
    # ISAM adds ~1.5M trainable params — adds 3D context for aorta, esophagus.
    if getattr(tc, "enable_isam", False):
        model.enable_3d()
        # Add ISAM params to optimizer as a separate low-LR group
        isam_params = [p for p in model.isam.parameters() if p.requires_grad]
        if isam_params:
            # ISAM starts from random init — use moderate LR, decay with poly
            isam_lr = tc.base_lr * getattr(tc, "isam_lr_scale", 2.0)
            optimizer.add_param_group({
                "params":       isam_params,
                "lr":           isam_lr,
                "weight_decay": 1e-4,
                "name":         "isam",
            })
            n_isam = sum(p.numel() for p in isam_params)
            logger.info(f"[ISAM] 3D inter-slice aggregation ENABLED")
            logger.info(f"  ISAM params: {n_isam/1e6:.2f}M  lr={isam_lr:.2e}")
            logger.info(f"  Total trainable now: "
                        f"{model.trainable_param_count()/1e6:.2f}M")

    for epoch in range(start_epoch, tc.epochs):

        # ── Optional encoder unfreeze ─────────────────────────────────────────
        # Only fires if --unfreeze_epoch N was explicitly passed.
        # Default is None → never fires.
        # Safe: checks that we haven't already unfrozen (avoids double-unfreeze
        # when resuming a checkpoint that was saved after an unfreeze).
        unfreeze_ep = getattr(tc, "unfreeze_epoch", None)
        if (unfreeze_ep is not None and epoch == unfreeze_ep
                and not getattr(tc, "_unfrozen", False)):
            tc._unfrozen = True
            logger.info("=" * 60)
            logger.info(f"[Unfreeze] Epoch {epoch+1} — unfreezing encoder stages 2+3")
            unfreeze_params = []
            for stage_idx in [2, 3]:
                for p in model.encoder_stages[stage_idx].parameters():
                    if not p.requires_grad:
                        p.requires_grad_(True)
                        unfreeze_params.append(p)
            if unfreeze_params:
                optimizer.add_param_group({
                    "params":       unfreeze_params,
                    "lr":           5e-6,          # 20× lower than decoder lr
                    "weight_decay": 1e-4,
                    "name":         "enc_unfreeze",
                })
                n = sum(p.numel() for p in unfreeze_params)
                logger.info(f"  Unfrozen {n/1e6:.2f}M params at lr=5e-6")
            logger.info("=" * 60)

        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        meter     = AverageMeter()
        nan_count = 0
        lr        = (scheduler.step(epoch)
                     if hasattr(scheduler, "step") else tc.base_lr)

        for step, batch in enumerate(train_loader):
            imgs   = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            # Shape check on first step only (not every step)
            if step == 0 and epoch == start_epoch:
                assert imgs.dim() == 4 and imgs.shape[1] == 3, \
                    f"Bad image shape: {imgs.shape} — expected [B,3,H,W]"
                assert labels.dim() == 3, \
                    f"Bad label shape: {labels.shape} — expected [B,H,W]"
                logger.info(f"  Input shape: {list(imgs.shape)}  "
                            f"Label shape: {list(labels.shape)}")

            with autocast(enabled=tc.use_amp):
                logits, aux_list = model(imgs, return_aux=True)
                loss_main = criterion(logits, labels)
                # Deep supervision: weighted sum of aux losses (weights decay with depth)
                # aux0 (deepest) weight 0.2, aux1 weight 0.3, aux2 weight 0.5
                aux_weights = [0.2, 0.3, 0.5]
                loss_aux  = sum(w * criterion(a, labels)
                                for w, a in zip(aux_weights, aux_list))
                loss = loss_main + getattr(tc, "aux_weight", 0.2) * loss_aux

            # ── NaN guard ─────────────────────────────────────────────────────
            if not loss.isfinite():
                nan_count += 1
                nan_total += 1
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if nan_count <= 5 or nan_count % 100 == 0:
                    logger.warning(
                        f"  ⚠ NaN loss  epoch {epoch+1} step {step}  "
                        f"(total skipped: {nan_total})")
                continue

            if tc.accumulation_steps > 1:
                loss = loss / tc.accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % tc.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), tc.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            meter.update(loss.item() * tc.accumulation_steps, imgs.size(0))

            if step % tc.log_interval == 0:
                logger.info(
                    f"Epoch {epoch+1}/{tc.epochs} "
                    f"Step {step}/{len(train_loader)} "
                    f"Loss={meter.avg:.4f}  LR={lr:.2e}"
                    + (f"  ⚠NaN×{nan_count}" if nan_count else "")
                )

        logger.info(
            f"Epoch {epoch+1} train loss={meter.avg:.4f}"
            + (f"  ⚠ skipped {nan_count} NaN batches" if nan_count else "")
        )

        # ── Validation every epoch ────────────────────────────────────────────
        # Full per-class breakdown every 10 epochs.
        # Every other epoch: summary + 5 worst classes (tracks small organs
        # without flooding the log).
        verbose = False  # worst class logging disabled
        if (epoch + 1) == tc.epochs or epoch == start_epoch:
            results = evaluate(model, val_loader, cfg, device, logger,
                           verbose=verbose)
            metric  = primary_metric(results, key)
            is_best = metric > best_metric
        else:
            metric  = best_metric
            is_best = False

        if is_best:
            best_metric = metric
            save_checkpoint(
                {"epoch":       epoch,
                 "model":       model.state_dict(),
                 "optimizer":   optimizer.state_dict(),
                 "best_metric": best_metric},
                out_dir, "best_model.pth",
            )

        # ── One-line epoch summary ─────────────────────────────────────────────
        logger.info(
            f"[Epoch {epoch+1:03d}/{tc.epochs}]  "
            f"loss={meter.avg:.4f}  "
            f"mean_dice={metric:.4f}  "
            f"best={best_metric:.4f}"
            + ("  ← BEST [saved]" if is_best else "")
            + (f"  ⚠NaN×{nan_count}" if nan_count else "")
        )

        # ── Periodic checkpoint ────────────────────────────────────────────────
        if (epoch + 1) % tc.save_interval == 0:
            save_checkpoint(
                {"epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "best_metric": best_metric},
                out_dir, f"epoch_{epoch+1:04d}.pth",
            )

    # Final save
    save_checkpoint(
        {"epoch": tc.epochs, "model": model.state_dict(),
         "best_metric": best_metric},
        out_dir, "final_model.pth",
    )
    logger.info(
        f"Training complete.  Best metric: {best_metric:.4f}  "
        f"Total NaN batches: {nan_total}"
    )
    return best_metric


# =========================================================================== #
#  Entry point
# =========================================================================== #

def main():
    args = parse_args()
    cfg  = config_from_dataset_name(args.dataset, args.data_root)
    cfg  = apply_overrides(args, cfg)

    log_dir = os.path.join(cfg.train.output_dir, cfg.train.experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    logger  = get_logger(
        "peft_umamba",
        os.path.join(log_dir, f"{args.stage}.log"),
    )

    logger.info("=" * 60)
    logger.info(" PEFT-UMamba Training")
    logger.info("=" * 60)
    logger.info(f"Dataset  : {cfg.data.dataset}  "
                f"({cfg.data.num_classes} classes, {cfg.data.modality})")
    logger.info(f"Data root: {cfg.data.data_root}")
    logger.info(f"Pretrained: {cfg.model.pretrained_path}")
    logger.info(f"Output   : {cfg.train.output_dir}/{cfg.train.experiment_name}")
    logger.info(f"Img size : {cfg.data.img_size}  "
                f"Batch: {cfg.train.batch_size}  "
                f"Epochs: {cfg.train.epochs}")
    logger.info(f"Loss: {cfg.train.loss}  "
                f"LR: {cfg.train.base_lr:.2e}  "
                f"K′={cfg.model.peft.supp_state_dim}  "
                f"supp_lr×{getattr(cfg.train, 'supp_lr_scale', 3.0)}")
    logger.info(f"GradClip: {cfg.train.gradient_clip}  "
                f"Warmup: {cfg.train.warmup_epochs} ep  "
                f"AMP: {cfg.train.use_amp}")

    set_seed(cfg.train.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # GPU knobs
    if device.type == "cuda":
        torch.backends.cudnn.benchmark        = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32        = True
        logger.info("[Opt] cudnn.benchmark=True  allow_tf32=True")

    # ── CRITICAL: Check mamba_ssm CUDA kernel availability ───────────────────
    # If HAS_MAMBA_CUDA=False every SSM scan falls back to pure PyTorch,
    # causing ~16× slowdown (9016 Python loop iters per forward pass).
    from models.model import HAS_MAMBA_CUDA
    if HAS_MAMBA_CUDA:
        logger.info("[Opt] mamba_ssm CUDA kernel: ✓ ACTIVE  (fast path)")
    else:
        logger.warning(
            "[Opt] mamba_ssm CUDA kernel: ✗ NOT FOUND\n"
            "      Training will use the vectorised PyTorch fallback (~3× slower).\n"
            "      To get full speed, install the CUDA kernel:\n"
            "        pip install mamba-ssm\n"
            "      or build from source:\n"
            "        pip install mamba-ssm --no-build-isolation"
        )

    # Build model
    model = build_model(cfg)

    # Run stage
    if args.stage == "mim_adapt":
        run_mim_adaptation(model, cfg, device, logger)

    elif args.stage == "finetune":
        # Resume is handled inside run_finetune() via cfg.train.resume
        # which loads both model weights AND optimizer state + start_epoch.
        # Do NOT call load_checkpoint here — that would overwrite optimizer
        # state with None and restart from epoch 0.
        run_finetune(model, cfg, device, logger)


if __name__ == "__main__":
    main()

import argparse
import csv
import dataclasses
import json
import math
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation on shared GPUs
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs.config import config_from_dataset_name
from data.dataset import build_loaders, predict_volume, set_amp_dtype
from models.model import build_model, HAS_MAMBA_CUDA
from peft_baselines import apply_peft
from utils.utils import (AverageMeter, DiceCELoss, SegMetrics, build_optimizer,
                         build_scheduler, count_params, deep_supervision_loss,
                         get_logger, instance_f1, load_checkpoint,
                         measure_inference_ms, save_checkpoint, set_seed)

ROOT = Path(__file__).resolve().parent
DEFAULT_CKPT = ROOT / "data" / "vmamba" / "vssmtiny_dp01_ckpt_epoch_292.pth"
METHODS = ["peft_umamba", "frozen", "lora", "full_ft", "mona", "lorand", "e3va"]


# =========================================================================== #
#  Arguments → config
# =========================================================================== #

def parse_args():
    p = argparse.ArgumentParser(description="PEFT-UMamba",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", required=True,
                   help="dataset701|dataset702|dataset703|dataset704|kvasir (or alias)")
    p.add_argument("--data_root", required=True)
    p.add_argument("--pretrained", default=str(DEFAULT_CKPT),
                   help="original VMamba-Tiny checkpoint (vssmtiny_dp01_ckpt_epoch_292.pth)")
    p.add_argument("--method", default="peft_umamba", choices=METHODS)
    p.add_argument("--seed", type=int, default=0)

    # PEFT-UMamba components / ablations
    p.add_argument("--k_prime", type=int, default=4, help="K′ supplementary dims")
    p.add_argument("--no_supp_scan", action="store_true", help="Table 5 B")
    p.add_argument("--no_scam", action="store_true", help="Table 5 C")
    p.add_argument("--scam_unordered", action="store_true",
                   help="control: same SCAM, but each slice is its own sequence "
                        "(isolates sequence aggregation from the GRU bottleneck)")
    p.add_argument("--no_skip_gate", action="store_true", help="Table 5 D")
    p.add_argument("--no_deep_sup", action="store_true", help="Table 5 E")
    p.add_argument("--supp_init", default="neighbourhood",
                   choices=["neighbourhood", "s4d"], help="s4d = Table 5 F")

    # baselines
    p.add_argument("--lora_rank", type=int, default=4)
    p.add_argument("--peft_bottleneck", type=int, default=16)

    # protocol overrides (leave unset to follow §4.2)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--amp_dtype", default="auto", choices=["auto", "fp16", "bf16"],
                   help="auto = bf16 for full fine-tuning (fp16 overflows when all "
                        "encoder weights train), fp16 otherwise")
    p.add_argument("--max_train_steps", type=int, help="smoke test: steps per epoch")
    p.add_argument("--iters_per_epoch", type=int,
                   help="volumes: iterations per epoch (default 250; 0 = one full pass)")
    p.add_argument("--skip_done", action="store_true",
                   help="exit immediately if this experiment already has results.json")

    p.add_argument("--output_dir", default=str(ROOT / "outputs"))
    p.add_argument("--exp_name")
    p.add_argument("--resume", default="", help="explicit path to last.pth only")
    return p.parse_args()


def make_config(args):
    cfg = config_from_dataset_name(args.dataset, args.data_root)
    m, pe, t = cfg.model, cfg.model.peft, cfg.train

    m.pretrained_path = args.pretrained
    pe.supp_state_dim = args.k_prime
    pe.supp_init = args.supp_init
    pe.no_skip_gate = args.no_skip_gate
    pe.lora_rank = args.lora_rank
    m.use_scam = not args.no_scam
    m.scam_unordered = args.scam_unordered
    t.deep_sup = not args.no_deep_sup

    pe.use_supplementary_scan = (args.method == "peft_umamba"
                                 and not args.no_supp_scan and args.k_prime > 0)
    pe.use_lora = args.method == "lora"
    m.freeze_encoder = args.method != "full_ft"

    if args.epochs is not None:
        t.epochs = args.epochs
    if args.batch_size is not None:
        t.batch_size = args.batch_size
    if args.lr is not None:
        t.base_lr = args.lr
    cfg.data.num_workers = args.num_workers
    t.eval_every = args.eval_every
    t.use_amp = not args.no_amp
    t.seed = args.seed
    t.max_train_steps = args.max_train_steps
    ipe = getattr(args, "iters_per_epoch", None)
    if ipe is not None:
        t.iters_per_epoch = ipe or None
    t.output_dir = args.output_dir

    if args.exp_name:
        t.experiment_name = args.exp_name
    else:
        tag = [cfg.data.dataset, args.method]
        if args.method == "peft_umamba":
            if args.k_prime != 4:
                tag.append(f"k{args.k_prime}")
            if args.no_supp_scan:
                tag.append("noSupp")
            if args.supp_init != "neighbourhood":
                tag.append("noNI")
        if args.no_scam:
            tag.append("noSCAM")
        if args.scam_unordered:
            tag.append("scamNoSeq")
        if args.no_skip_gate:
            tag.append("noGate")
        if args.no_deep_sup:
            tag.append("noDS")
        tag.append(f"s{args.seed}")
        t.experiment_name = "_".join(tag)
    return cfg


# =========================================================================== #
#  Evaluation
# =========================================================================== #

AMP_DTYPE = torch.float16


def _np(x):
    return x.numpy() if torch.is_tensor(x) else np.asarray(x)


@torch.no_grad()
def evaluate(model, val_loader, cfg, names, device, final=True):
    model.eval()
    dc, tc = cfg.data, cfg.train
    # HD95 (CPU distance transforms) only for the reported final evaluation
    metrics = SegMetrics(len(names), names, compute_hd95=dc.kind == "volume" and final,
                         compute_instance_f1=dc.instance_f1)
    amp = tc.use_amp and device.type == "cuda"
    if dc.kind == "volume":
        for item in val_loader:
            vol, gt = _np(item["image"]).astype(np.float32), _np(item["label"])
            pred = predict_volume(model, vol, dc.img_size, tc.batch_size, device, amp)
            sp = tuple(float(v) for v in _np(item["spacing"])) if "spacing" in item \
                else (dc.spacing,) * 3
            metrics.add_case(pred, gt, spacing=sp, case=item.get("case"))
    else:
        for batch in val_loader:
            x = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=amp):
                logits = model(x)
            pred = logits.argmax(1).cpu().numpy()
            gt = batch["label"].numpy()
            cases = batch.get("case", [None] * pred.shape[0])
            for b in range(pred.shape[0]):
                f1 = None
                if "instances" in batch:      # official CellSeg: F1 at native resolution
                    inst = batch["instances"][b].numpy()
                    up = F.interpolate(logits[b:b + 1].float(), size=inst.shape,
                                       mode="bilinear", align_corners=False)
                    f1 = instance_f1(up.argmax(1)[0].cpu().numpy(), None,
                                     gt_instances=inst)
                metrics.add_case(pred[b], gt[b], case=cases[b], inst_f1=f1)
    model.train()
    return metrics.compute()


# =========================================================================== #
#  Main
# =========================================================================== #

def main():
    args = parse_args()
    cfg = make_config(args)
    tc, dc = cfg.train, cfg.data
    out_dir = Path(tc.output_dir) / tc.experiment_name
    if args.skip_done and (out_dir / "results.json").exists():
        print(f"[skip] {tc.experiment_name} already finished")
        return
    # lock: several queues may share one run list (e.g. two queues per GPU)
    lock = out_dir / "RUNNING"
    if args.skip_done and lock.exists():
        try:
            os.kill(int(lock.read_text().split()[0]), 0)
            print(f"[skip] {tc.experiment_name} is running in another queue")
            return
        except (ValueError, IndexError, ProcessLookupError):
            pass                                   # stale lock: owner is gone
        except PermissionError:
            print(f"[skip] {tc.experiment_name} is running in another queue")
            return
    out_dir.mkdir(parents=True, exist_ok=True)
    lock.write_text(str(os.getpid()))
    log = get_logger("peft_umamba", str(out_dir / "train.log"))

    set_seed(tc.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    if not HAS_MAMBA_CUDA:
        log.warning("mamba_ssm CUDA kernel not found — the reference scan is very slow")

    # data first: it fixes num_classes from dataset.json
    train_loader, val_loader, names = build_loaders(cfg)

    model = build_model(cfg)
    if args.method in ("mona", "lorand", "e3va"):
        n = apply_peft(model.encoder, args.method, args.peft_bottleneck)
        log.info(f"[{args.method}] bottleneck={args.peft_bottleneck}  "
                 f"adaptation params={n/1e6:.3f}M")
    model = model.to(device)

    adapt_params = count_params(model.encoder)            # trainable inside encoder
    total_params = count_params(model, trainable_only=False)
    train_params = count_params(model)
    with open(out_dir / "config.json", "w") as f:
        json.dump({"args": vars(args), "cfg": dataclasses.asdict(cfg)}, f, indent=1)
    log.info(f"Experiment {tc.experiment_name}  method={args.method}  "
             f"dataset={dc.dataset}  classes={len(names)}")
    log.info(f"Params: total {total_params/1e6:.2f}M  trainable {train_params/1e6:.3f}M  "
             f"encoder-adaptation {adapt_params/1e6:.3f}M")
    log.info(f"Protocol: epochs {tc.epochs}  batch {tc.batch_size}  lr {tc.base_lr}  "
             f"wd {tc.weight_decay}  warmup {tc.warmup_epochs}ep  poly {tc.poly_power}  "
             f"deep_sup {tc.deep_sup} {tc.deep_sup_weights}  ordered={dc.ordered}  "
             f"iters/epoch {len(train_loader)}")

    criterion = DiceCELoss(len(names), tc.ce_weight, tc.dice_weight)
    optimizer = build_optimizer(model, tc.base_lr, tc.weight_decay)
    steps = len(train_loader) if tc.max_train_steps is None else min(
        len(train_loader), tc.max_train_steps)
    scheduler = build_scheduler(optimizer, tc.epochs * steps,
                                tc.warmup_epochs * steps, tc.poly_power)
    amp = tc.use_amp and device.type == "cuda"
    global AMP_DTYPE
    AMP_DTYPE = torch.bfloat16 if (args.amp_dtype == "bf16" or (
        args.amp_dtype == "auto" and args.method == "full_ft")) else torch.float16
    set_amp_dtype(AMP_DTYPE)
    log.info(f"mixed precision: {'off' if not amp else AMP_DTYPE}")
    scaler = torch.amp.GradScaler("cuda", enabled=amp and AMP_DTYPE == torch.float16)

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, args.resume, optimizer, scheduler)
        log.info(f"Resumed from {args.resume} at epoch {start_epoch}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(start_epoch, tc.epochs):
        model.train()
        meter, nan_steps, t0 = AverageMeter(), 0, time.time()
        for step, batch in enumerate(train_loader):
            if step >= steps:
                break
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=amp):
                logits, aux = model(x, return_aux=True)
                loss = deep_supervision_loss(criterion, logits, aux, y,
                                             tc.deep_sup_weights, tc.deep_sup)
            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                nan_steps += 1
                scheduler.step()
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], tc.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            meter.update(loss.item(), x.shape[0])

        log.info(f"epoch {epoch+1:3d}/{tc.epochs}  loss {meter.avg:.4f}  "
                 f"lr {optimizer.param_groups[0]['lr']:.2e}  {time.time()-t0:.0f}s"
                 + (f"  skipped {nan_steps} non-finite steps" if nan_steps else ""))

        last = epoch + 1 == tc.epochs
        if (epoch + 1) % tc.eval_every == 0 and not last:
            r = evaluate(model, val_loader, cfg, names, device, final=False)
            log.info(f"  [monitor] val dice {r['dice']:.4f}")
            save_checkpoint({"epoch": epoch + 1, "model": model.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "scheduler": scheduler.state_dict()},
                            str(out_dir), "last.pth")

    # ── final evaluation (reported) ───────────────────────────────────────────
    peak_gb = (torch.cuda.max_memory_allocated(device) / 1024 ** 3
               if device.type == "cuda" else float("nan"))
    res = evaluate(model, val_loader, cfg, names, device)
    infer_ms = measure_inference_ms(model, dc.img_size, device)
    save_checkpoint({"epoch": tc.epochs, "model": model.state_dict()},
                    str(out_dir), "final.pth")

    summary = {
        "dataset": dc.dataset, "method": args.method, "exp": tc.experiment_name,
        "seed": tc.seed, "k_prime": cfg.model.peft.supp_state_dim
        if cfg.model.peft.use_supplementary_scan else 0,
        "total_M": round(total_params / 1e6, 3),
        "trainable_M": round(train_params / 1e6, 3),
        "adapt_M": round(adapt_params / 1e6, 3),
        "train_mem_GB": round(peak_gb, 2), "infer_ms": round(infer_ms, 2),
        "dice": res["dice"], "dice_std_cases": res["dice_std"],
        "hd95": res.get("hd95", float("nan")),
        "instance_f1": res.get("instance_f1", float("nan")),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump({"summary": summary, "metrics": res}, f, indent=1)
    csv_path = Path(tc.output_dir) / "summary.csv"
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary))
        if new:
            w.writeheader()
        w.writerow(summary)

    log.info("=" * 70)
    log.info(f"FINAL  dice {res['dice']:.4f} ± {res['dice_std']:.4f} (across cases)"
             + (f"  hd95 {res['hd95']:.2f}" if "hd95" in res else "")
             + (f"  instF1 {res['instance_f1']:.4f}" if "instance_f1" in res else ""))
    for k, v in res.items():
        if k.startswith("dice_") and k != "dice_std":
            log.info(f"  {k[5:]:<24s} dice {v:.4f}"
                     + (f"  hd95 {res.get('hd95_' + k[5:], float('nan')):.2f}"
                        if "hd95" in res else ""))
    log.info(f"[Table6] trainable {train_params/1e6:.3f}M  adapt {adapt_params/1e6:.3f}M  "
             f"mem {peak_gb:.2f}GB  infer {infer_ms:.2f}ms")
    lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

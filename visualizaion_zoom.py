"""
visualize_table6.py — qualitative comparison of the Table 6 adaptation
strategies (Reviewer 1, Comment 1.3).

Loads the FINAL checkpoint of every method for one dataset and seed, predicts
the same validation cases, and draws one row per case:
    image | ground truth | frozen | LoRA | full FT | Mona | LoRand | E3VA | PEFT-UMamba
with the per-case DSC above each prediction.

Cases are chosen objectively from PEFT-UMamba's own per-case DSC ranking
(25th, 50th and 75th percentile, plus the worst case), so the figure is
representative and includes a failure case rather than only the best examples.

    python visualize_table6.py --dataset kvasir     --data_root /path/kvasir
    python visualize_table6.py --dataset dataset702 --data_root /path/Dataset702_AbdomenMR
Output: outputs/figures/table6_<dataset>_s<seed>.png / .pdf
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs.config import resolve_dataset_key
from data.dataset import IMAGENET_MEAN, IMAGENET_STD, build_loaders, predict_volume
from models.model import build_model
from peft_baselines import apply_peft
from train import make_config
from utils.utils import dice_per_class

ROOT = Path(__file__).resolve().parent
METHODS = [("frozen", "Frozen enc."), ("lora", "LoRA"), ("full_ft", "Full FT"),
           ("mona", "Mona"), ("lorand", "LoRand"), ("e3va", "E3VA"),
           ("peft_umamba", "PEFT-UMamba")]


def load_run(exp_dir: Path, device):
    saved = json.loads((exp_dir / "config.json").read_text())["args"]
    args = argparse.Namespace(**saved)
    args.num_workers = 0
    cfg = make_config(args)
    _, val_loader, names = build_loaders(cfg)
    model = build_model(cfg)
    if args.method in ("mona", "lorand", "e3va"):
        apply_peft(model.encoder, args.method, args.peft_bottleneck)
    state = torch.load(exp_dir / "final.pth", map_location="cpu")["model"]
    model.load_state_dict(state)
    return model.to(device).eval(), cfg, val_loader.dataset, names


@torch.no_grad()
def predict_case(model, cfg, ds, i, device):
    """Returns (display image [H,W] or [H,W,3], gt, pred, case DSC)."""
    item = ds[i]
    if cfg.data.kind == "volume":
        vol, gt = item["image"], item["label"]
        pred = predict_volume(model, vol, cfg.data.img_size, cfg.train.batch_size,
                              device, use_amp=device.type == "cuda")
        img = vol
    else:
        x = item["image"][None].to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            pred = model(x).argmax(1)[0].cpu().numpy()
        gt = item["label"].numpy()
        img = (item["image"] * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
        img = img.permute(1, 2, 0).numpy()
    dsc = float(np.nanmean(dice_per_class(pred, gt, cfg.data.num_classes)))
    return img, gt, pred, dsc


def pick_cases(dscs):
    order = np.argsort(dscs)
    n = len(order)
    picks = [order[int(round(q * (n - 1)))] for q in (0.75, 0.5, 0.25)] + [order[0]]
    seen, out = set(), []
    for p in picks:
        if p not in seen:
            seen.add(p)
            out.append(int(p))
    return out


def best_slice(gt_vol):
    """Axial slice with the most foreground classes (ties: most foreground)."""
    scores = [(len(np.unique(s)) - 1, (s > 0).sum()) for s in gt_vol]
    return int(max(range(len(scores)), key=lambda k: scores[k]))


def overlay(ax, img, lbl, num_classes):
    import matplotlib.pyplot as plt
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
    cmap = plt.get_cmap("tab20", max(num_classes, 2))
    rgba = cmap(lbl % cmap.N)
    rgba[..., 3] = np.where(lbl > 0, 0.55, 0.0)
    ax.imshow(rgba)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outputs", default=str(ROOT / "outputs"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = resolve_dataset_key(args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = [(m, label, Path(args.outputs) / f"{key}_{m}_s{args.seed}") for m, label in METHODS]
    runs = [r for r in runs if (r[2] / "final.pth").exists()]
    if not runs or runs[-1][0] != "peft_umamba":
        raise SystemExit("PEFT-UMamba final.pth not found for this dataset/seed — "
                         "train it first (train.py --method peft_umamba).")

    # 1. choose cases from PEFT-UMamba's per-case DSC
    model, cfg, ds, names = load_run(runs[-1][2], device)
    dscs = [predict_case(model, cfg, ds, i, device)[3] for i in range(len(ds))]
    cases = pick_cases(np.array(dscs))
    del model
    print(f"selected validation cases {cases} (PEFT-UMamba DSC "
          f"{[round(dscs[c], 3) for c in cases]})")

    # 2. predict those cases with every method
    preds, base = {}, {}
    for m, label, exp in runs:
        model, cfg, ds, names = load_run(exp, device)
        for c in cases:
            img, gt, pred, dsc = predict_case(model, cfg, ds, c, device)
            base[c] = (img, gt)
            preds[(m, c)] = (pred, dsc)
        del model
        torch.cuda.empty_cache()
        print(f"  {label}: done")

    # 3. figure
    ncol = 2 + len(runs)
    fig, axes = plt.subplots(len(cases), ncol, figsize=(1.9 * ncol, 1.95 * len(cases)),
                             squeeze=False)
    for r, c in enumerate(cases):
        img, gt = base[c]
        sl = best_slice(gt) if gt.ndim == 3 else None
        show = (lambda a: a[sl]) if sl is not None else (lambda a: a)
        ax = axes[r, 0]
        ax.imshow(show(img), cmap="gray" if show(img).ndim == 2 else None)
        ax.set_xticks([]); ax.set_yticks([])
        overlay(axes[r, 1], show(img), show(gt), len(names))
        for j, (m, label, _) in enumerate(runs):
            pred, dsc = preds[(m, c)]
            overlay(axes[r, 2 + j], show(img), show(pred), len(names))
            axes[r, 2 + j].set_title(f"{dsc:.3f}", fontsize=8, pad=2)
        if r == 0:
            axes[0, 0].set_xlabel("")
        axes[r, 0].set_ylabel(f"case {r + 1}", fontsize=8)
    for j, t in enumerate(["Image", "Ground truth"] + [lab for _, lab, _ in runs]):
        axes[-1, j].set_xlabel(t, fontsize=8)
    fig.tight_layout(pad=0.3, w_pad=0.2, h_pad=0.6)

    out = Path(args.outputs) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"table6_{key}_s{args.seed}"
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    print(f"saved {stem}.png / .pdf")


if __name__ == "__main__":
    main()

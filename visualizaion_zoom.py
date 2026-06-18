"""
visualization.py — IEEE-style publication figures for PEFT-UMamba (AMOS22).
FIXED VERSION: larger fonts, no text overlap, better readability at low zoom.
"""

import argparse, csv, os, re, sys, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, str(Path(__file__).parent))

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not found — inference figures skipped")

# =========================================================================== #
#  IEEE / TMI style — FIXED: larger fonts, more padding
# =========================================================================== #

TMI_1COL = 3.50
TMI_2COL = 7.16
DPI      = 300

plt.rcParams.update({
    "font.family":           "serif",
    "font.serif":            ["Times New Roman", "DejaVu Serif"],
    "font.size":             13,
    "axes.labelsize":        13,
    "axes.titlesize":        14,
    "axes.linewidth":        1.0,
    "xtick.labelsize":       12,
    "ytick.labelsize":       12,
    "xtick.major.width":     1.0,
    "ytick.major.width":     1.0,
    "xtick.major.size":      5,
    "ytick.major.size":      5,
    "legend.fontsize":       11,
    "legend.title_fontsize": 12,
    "legend.framealpha":     0.95,
    "legend.edgecolor":      "0.4",
    "legend.borderpad":      0.7,
    "lines.linewidth":       2.0,
    "patch.linewidth":       0.8,
    "figure.dpi":            DPI,
    "savefig.dpi":           DPI,
    "savefig.bbox":          "tight",
    "savefig.pad_inches":    0.05,
    "pdf.fonttype":          42,
    "ps.fonttype":           42,
})

# =========================================================================== #
#  AMOS22 class metadata
# =========================================================================== #

AMOS_CLASSES = [
    "background",
    "spleen", "right_kidney", "left_kidney", "gallbladder",
    "esophagus", "liver", "stomach", "aorta",
    "inferior_vena_cava", "pancreas",
    "right_adrenal_gland", "left_adrenal_gland",
    "duodenum", "bladder", "prostate_uterus",
]

FINAL_DICE = {
    "spleen":             0.9148,
    "right_kidney":       0.7029,
    "left_kidney":        0.7812,
    "gallbladder":        0.6103,
    "esophagus":          0.7786,
    "liver":              0.5788,
    "stomach":            0.3200,
    "aorta":              0.3031,
    "inferior_vena_cava": 0.5890,
    "pancreas":           0.5380,
    "right_adrenal_gland":0.6223,
    "left_adrenal_gland": 0.3400,
    "duodenum":           0.6720,
    # bladder and prostate excluded — sparse validation cases
    # "bladder": 1.0000,
    # "prostate_uterus": 1.0000,
}

FINAL_HD95 = {
    "spleen":             37.84,
    "right_kidney":       87.94,
    "left_kidney":        59.51,
    "gallbladder":        46.21,
    "esophagus":          19.20,
    "liver":              22.06,
    "stomach":            10.99,
    "aorta":              11.73,
    "inferior_vena_cava": 12.47,
    "pancreas":           26.23,
    "right_adrenal_gland":61.43,
    "left_adrenal_gland": 47.55,
    "duodenum":           77.67,
    # "bladder":            float("nan"),
    # "prostate_uterus":    float("nan"),
}

AMOS_HEX = [
    "#000000","#E6194B","#3CB44B","#4363D8","#F58231",
    "#911EB4","#42D4F4","#F032E6","#BFEF45","#FABED4",
    "#469990","#DCBEFF","#9A6324","#FFFAC8","#800000","#AAFFC3",
]

# =========================================================================== #
#  Shared helpers
# =========================================================================== #

def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))


def save_fig(fig, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}")
    plt.close(fig)
    print(f"  OK  {Path(path).name}.{{pdf,png}}")


def ax_clean(ax, title="", ylabel=""):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
        sp.set_color("#AAAAAA")
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10.5, rotation=90,
                      labelpad=5, va="center")


def organ_overlay(mask):
    rgba = np.zeros((*mask.shape, 4), np.float32)
    for c in range(1, len(AMOS_HEX)):
        region = mask == c
        if not region.any():
            continue
        r, g, b = hex_rgb(AMOS_HEX[c])
        border  = region & ~binary_erosion(region, iterations=1)
        rgba[region] = [r, g, b, 0.50]
        rgba[border] = [r, g, b, 0.95]
    return rgba


def error_map(gt, pred):
    rgba = np.zeros((*gt.shape, 4), np.float32)
    fg_gt   = gt   > 0
    fg_pred = pred > 0
    rgba[fg_gt  & fg_pred]  = [0.05, 0.78, 0.18, 0.70]
    rgba[~fg_gt & fg_pred]  = [0.95, 0.12, 0.12, 0.80]
    rgba[fg_gt  & ~fg_pred] = [0.12, 0.32, 0.95, 0.80]
    return rgba


def dice_2d(p, g):
    inter = int((p & g).sum())
    denom = int(p.sum() + g.sum())
    return 2 * inter / denom if denom else 1.0


def best_fg_slice(gt, cls_id=None):
    if cls_id is not None:
        counts = [(gt[z] == cls_id).sum() for z in range(gt.shape[0])]
    else:
        counts = [(gt[z] > 0).sum() for z in range(gt.shape[0])]
    return int(np.argmax(counts))


def display_vol(img):
    lo, hi = img.min(), img.max()
    return (img - lo) / (hi - lo + 1e-8)


# =========================================================================== #
#  NIfTI I/O
# =========================================================================== #

def load_nifti(path):
    import nibabel as nib
    nii  = nib.load(str(path))
    data = nii.get_fdata(dtype=np.float32)
    while data.ndim > 3:
        data = data[..., 0]
    return np.transpose(data, (2, 1, 0)), nii.affine


def norm_mri(v):
    mu, s = v.mean(), v.std() + 1e-8
    return ((np.clip((v - mu) / s, -4, 4) + 4) / 8).astype(np.float32)


# =========================================================================== #
#  Model inference
# =========================================================================== #

def run_inference(ckpt, data_root, device):
    from configs.config import config_from_dataset_name
    from models.model import build_model
    from utils.utils import load_checkpoint
    from data.dataset import NiftiSegDataset, sliding_window_predict

    cfg = config_from_dataset_name("dataset702", data_root)
    cfg.model.freeze_encoder = False
    model = build_model(cfg)
    load_checkpoint(model, ckpt)
    model = model.to(device).eval()

    ds = NiftiSegDataset(data_root, "dataset702",
                         split="val", mode="volume",
                         img_size=224, augment=False)
    results = []
    print(f"  Inference on {len(ds.pairs)} val volumes ...")
    t0 = __import__("time").time()

    for idx in range(len(ds.pairs)):
        batch    = ds[idx]
        img_vol  = batch["image"]
        gt_vol   = batch["label"].astype(np.int64)
        case_id  = batch["case"]
        pred_vol = sliding_window_predict(img_vol, model, 224, 16, device, "mri")

        per_class = {}
        for c in range(1, 16):
            p = pred_vol == c
            g = gt_vol   == c
            inter = int((p & g).sum())
            denom = int(p.sum() + g.sum())
            per_class[c] = 2*inter/denom if denom else (1.0 if not g.any() else 0.0)

        fg_present = [c for c in range(1, 16) if (gt_vol == c).any()]
        mean_dsc   = float(np.mean([per_class[c] for c in fg_present])) if fg_present else 0.0

        results.append(dict(case_id=case_id, img_vol=img_vol, gt_vol=gt_vol,
                            pred_vol=pred_vol, per_class=per_class, mean_dsc=mean_dsc))
        elapsed = __import__("time").time() - t0
        print(f"    [{idx+1:3d}/{len(ds.pairs)}]  {case_id:<20s}  DSC={mean_dsc:.4f}  ({elapsed:.0f}s)")

    return results


# =========================================================================== #
#  Fig 1 — Per-class Dice bar chart  FIXED: taller, no HD95 overlap
# =========================================================================== #

def fig_per_class_dice(output_dir, dsc_threshold=0.60, runtime_dice=None):
    dice  = runtime_dice if runtime_dice else FINAL_DICE
    names = [n for n in AMOS_CLASSES[1:] if n in dice]
    vals  = np.array([dice[n] for n in names])
    order = np.argsort(vals)[::-1]
    names = [names[i] for i in order]
    vals  = vals[order]

    def bar_color(d):
        if d >= 0.80: return "#2166AC"
        if d >= 0.60: return "#4DAC26"
        if d >= 0.40: return "#F4A582"
        return "#D6604D"

    colors = [bar_color(d) for d in vals]
    # FIXED: taller figure
    fig, ax = plt.subplots(figsize=(TMI_2COL, TMI_2COL * 0.58))

    y    = np.arange(len(names))
    bars = ax.barh(y, vals, color=colors, edgecolor="white",
                   linewidth=0.3, height=0.72)

    # FIXED: fontsize 11, pushed right
    for v, b in zip(vals, bars):
        ax.text(v + 0.015,
                b.get_y() + b.get_height() / 2,
                f"{v:.3f}", va="center", ha="left",
                fontsize=11)

    ax.axvline(dsc_threshold, color="#D6604D", lw=1.2, ls="--",
               label=f"Threshold = {dsc_threshold:.2f}")
    mean_dsc = float(np.mean(vals))
    ax.axvline(mean_dsc, color="#2166AC", lw=1.2, ls="-.",
               label=f"Mean DSC = {mean_dsc:.4f}")

    ax.set_yticks(y)
    short = {
        "right_kidney":        "R. Kidney",
        "left_kidney":         "L. Kidney",
        "right_adrenal_gland": "R. Adrenal",
        "left_adrenal_gland":  "L. Adrenal",
        "inferior_vena_cava":  "IVC",
        "prostate_uterus":     "Prostate*",
        "gallbladder":         "Gallbladder",
        "esophagus":           "Esophagus",
        "pancreas":            "Pancreas",
        "duodenum":            "Duodenum",
        "bladder":             "Bladder",
        "stomach":             "Stomach",
        "spleen":              "Spleen",
        "liver":               "Liver",
        "aorta":               "Aorta",
    }
    ax.set_yticklabels([short.get(n, n.replace("_"," ").title()) for n in names], fontsize=11)
    ax.set_xlabel("Dice Similarity Coefficient", fontsize=12)
    ax.set_xlim(0.18, 1.18)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.grid(axis="x", which="major", lw=0.3, alpha=0.4)
    ax.grid(axis="x", which="minor", lw=0.15, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    legend_elems = [
        mpatches.Patch(color="#2166AC", label="DSC >= 0.80"),
        mpatches.Patch(color="#4DAC26", label="DSC >= 0.60"),
        mpatches.Patch(color="#F4A582", label="DSC >= 0.40"),
        mpatches.Patch(color="#D6604D", label="DSC < 0.40"),
        plt.Line2D([0],[0], color="#D6604D", lw=1.2, ls="--",
                   label=f"Threshold {dsc_threshold:.2f}"),
        plt.Line2D([0],[0], color="#2166AC", lw=1.2, ls="-.",
                   label=f"Mean {mean_dsc:.4f}"),
    ]
    ax.legend(handles=legend_elems, loc="lower right",
              fontsize=10.5, framealpha=0.95, ncol=2)
    ax.set_title("AMOS22 16-Class MRI Segmentation — Per-Class DSC",
                 fontsize=13, fontweight="bold", pad=8)

    # FIXED: HD95 right axis — compact numbers only, no "HD95=" prefix overlap
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(y)
    hd_labels = [
        f"{FINAL_HD95.get(n, float('nan')):.0f}"
        if not np.isnan(FINAL_HD95.get(n, float("nan"))) else "--"
        for n in names
    ]
    ax2.set_yticklabels(hd_labels, fontsize=10, color="#555555", fontweight="bold")
    ax2.set_ylabel("HD95 (mm)", fontsize=11, color="#555555", labelpad=6, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_linewidth(0.5)
    ax2.invert_yaxis()

    save_fig(fig, os.path.join(output_dir, "fig1_per_class_dice"))


# =========================================================================== #
#  Fig 2 — Per-case overlay
# =========================================================================== #

def fig_case_overlay(r, output_dir, n_slices=3):
    fg_cls = [c for c in range(1, 16)
              if (r["gt_vol"] == c).any() or (r["pred_vol"] == c).any()]
    if not fg_cls:
        return

    fg_per_z = [(r["gt_vol"][z] > 0).sum() for z in range(r["gt_vol"].shape[0])]
    top_z    = np.argsort(fg_per_z)[::-1][:n_slices * 4]
    top_z    = sorted(top_z)
    step     = max(1, len(top_z) // n_slices)
    z_ids    = [top_z[i * step] for i in range(n_slices)][:n_slices]

    legend_cls = [c for c in fg_cls
                  if r["per_class"].get(c, 0) > 0.6 and c < len(AMOS_HEX)]
    n_leg_rows = max(1, (len(legend_cls) + 2) // 3)
    leg_h_in   = 0.35 * n_leg_rows + 0.30

    panel_h = TMI_2COL / 4
    fig_h   = n_slices * panel_h + 0.20 + leg_h_in
    fig     = plt.figure(figsize=(TMI_2COL, fig_h))

    leg_frac = leg_h_in / fig_h
    gs = gridspec.GridSpec(n_slices, 4, figure=fig,
                           wspace=0.015, hspace=0.02,
                           top=0.88, bottom=leg_frac + 0.01,
                           left=0.01, right=0.99)

    fig.suptitle(f"Case: {r['case_id']}   mean DSC = {r['mean_dsc']:.4f}",
                 fontsize=13, fontweight="bold", y=0.96)
    col_titles = ["MRI Input", "Ground Truth", "Prediction", "Error Map"]

    for row, z in enumerate(z_ids):
        img_d   = display_vol(r["img_vol"][z])
        gt_s    = r["gt_vol"][z]
        pred_s  = r["pred_vol"][z]
        img_rgb = np.stack([img_d] * 3, -1)

        for col, (base, ov) in enumerate([
            (img_rgb, None),
            (img_rgb, organ_overlay(gt_s)),
            (img_rgb, organ_overlay(pred_s)),
            (img_rgb, error_map(gt_s, pred_s)),
        ]):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(base, aspect="auto", interpolation="lanczos")
            if ov is not None:
                ax.imshow(ov, aspect="auto", interpolation="none")
            ax_clean(ax,
                     title=col_titles[col] if row == 0 else "",
                     ylabel=f"z = {z}" if col == 0 else "")
            if col == 2:
                sl_dices = [dice_2d(pred_s == c, gt_s == c)
                            for c in range(1, 16)
                            if (gt_s == c).any() or (pred_s == c).any()]
                if sl_dices:
                    # FIXED: white text, black bg, bigger font
                    ax.text(0.97, 0.03, f"DSC={np.mean(sl_dices):.3f}",
                            transform=ax.transAxes, fontsize=11,
                            ha="right", va="bottom", color="#FFFFFF",
                            fontweight="bold",
                            bbox=dict(facecolor="#000000", alpha=0.75,
                                      pad=3, boxstyle="round,pad=0.4"))

    if legend_cls:
        handles = [
            mpatches.Patch(
                color=hex_rgb(AMOS_HEX[c]),
                label=f"{AMOS_CLASSES[c].replace('_',' ')}  {r['per_class'].get(c,0):.3f}",
            )
            for c in sorted(legend_cls)
        ]
        fig.legend(handles=handles, loc="upper left",
                   bbox_to_anchor=(0.01, leg_frac + 0.005),
                   bbox_transform=fig.transFigure,
                   ncol=3, fontsize=10, framealpha=0.95,
                   title="Organ", title_fontsize=11,
                   edgecolor="0.5", borderaxespad=0.2)

    fig.legend(handles=[
        mpatches.Patch(color=(0.05, 0.78, 0.18), label="True Pos"),
        mpatches.Patch(color=(0.95, 0.12, 0.12), label="False Pos"),
        mpatches.Patch(color=(0.12, 0.32, 0.95), label="False Neg"),
    ], loc="upper right",
    bbox_to_anchor=(0.99, leg_frac + 0.005),
    bbox_transform=fig.transFigure,
    ncol=1, fontsize=10, framealpha=0.95, edgecolor="0.5")

    out = os.path.join(output_dir, "cases", f"case_{r['case_id']}")
    save_fig(fig, out)


# =========================================================================== #
#  Fig 3 — Learning curve
# =========================================================================== #

def fig_learning_curve(log_path, output_dir):
    if not Path(log_path).exists():
        print(f"  [SKIP] log not found: {log_path}")
        return

    text   = Path(log_path).read_text(errors="replace")
    ep_loss, ep_dice, ep_best = {}, {}, {}

    for m in re.finditer(r"Epoch\s+(\d+)\s+train\s+loss=([\d.]+)", text):
        ep_loss[int(m.group(1))] = float(m.group(2))
    for m in re.finditer(r"\[Epoch\s*(\d+)/\d+\].*?mean_dice=([\d.]+).*?best=([\d.]+)", text):
        ep = int(m.group(1))
        ep_dice[ep] = float(m.group(2))
        ep_best[ep] = float(m.group(3))

    if not ep_loss and not ep_dice:
        print("  [SKIP] no parseable epoch data in log")
        return

    epochs = sorted(set(list(ep_loss) + list(ep_dice)))
    losses = np.array([ep_loss.get(e, np.nan) for e in epochs])
    dices  = np.array([ep_dice.get(e, np.nan) for e in epochs])
    bests  = np.array([ep_best.get(e, np.nan) for e in epochs])
    epochs = np.array(epochs)

    blue, red, grn = "#2166AC", "#D6604D", "#4DAC26"
    fig, (ax_l, ax_d) = plt.subplots(1, 2, figsize=(TMI_2COL, TMI_2COL * 0.50))

    valid = ~np.isnan(losses)
    if valid.any():
        ax_l.plot(epochs[valid], losses[valid], color=blue, lw=2.0, label="Train loss")
    ax_l.axvline(18, color=red, lw=1.0, ls=":", alpha=0.75)
    ax_l.text(19, 0.9, "resume\nep18", fontsize=11, color=red, va="top")
    ax_l.set_xlabel("Epoch"); ax_l.set_ylabel("Training Loss")
    ax_l.set_title("(a) Training Loss", fontweight="bold")
    ax_l.spines["top"].set_visible(False)
    ax_l.spines["right"].set_visible(False)
    ax_l.grid(lw=0.3, alpha=0.4)

    valid = ~np.isnan(dices)
    if valid.any():
        ax_d.plot(epochs[valid], dices[valid], color=blue, lw=2.0, label="Val Dice")
    valid_b = ~np.isnan(bests)
    if valid_b.any():
        ax_d.plot(epochs[valid_b], bests[valid_b], color=grn, lw=1.0,
                  ls="--", label="Best so far")
    if valid.any():
        best_ep  = epochs[valid][np.nanargmax(dices[valid])]
        best_val = np.nanmax(dices[valid])
        ax_d.axvline(best_ep, color=grn, lw=0.8, ls="--", alpha=0.65)
        ax_d.annotate(f"best={best_val:.4f}\n(ep{best_ep})",
                      xy=(best_ep, best_val), xytext=(5, -18),
                      textcoords="offset points", fontsize=11, color=grn)

    ax_d.axhline(0.60, color=red, lw=1.0, ls=":", alpha=0.7, label="DSC=0.60")
    ax_d.axvline(18, color=red, lw=1.0, ls=":", alpha=0.75)
    ax_d.text(19, 0.10, "resume\nep18", fontsize=11, color=red)
    ax_d.set_ylim(0, 1.0)
    ax_d.set_xlabel("Epoch"); ax_d.set_ylabel("Val Mean DSC")
    ax_d.set_title("(b) Validation Dice", fontweight="bold")
    ax_d.legend(fontsize=11, loc="lower right", framealpha=0.95)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)
    ax_d.grid(lw=0.3, alpha=0.4)

    fig.subplots_adjust(wspace=0.40)
    save_fig(fig, os.path.join(output_dir, "fig3_learning_curve"))


# =========================================================================== #
#  Fig 4b — Best-organ spotlight
# =========================================================================== #

def fig_best_organ_spotlight(results, output_dir):
    best_organs = [
        ("spleen",              1,  0.9148),
        ("esophagus",           5,  0.7786),
        ("left_kidney",         3,  0.7812),
        ("duodenum",           13,  0.6720),
        ("right_adrenal_gland",11,  0.6223),
    ]
    n_organs, n_cases = len(best_organs), 2

    fig = plt.figure(figsize=(TMI_2COL, TMI_2COL * 1.62))
    fig.suptitle("High-Performance Organ Segmentation — Best / Median Cases",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(n_organs * n_cases, 4, figure=fig,
                           wspace=0.015, hspace=0.04,
                           top=0.88, bottom=0.07, left=0.13, right=0.96)
    col_titles = ["MRI Input", "Ground Truth", "Prediction", "Error Map"]

    for oi, (organ_name, cls_id, ep150_dsc) in enumerate(best_organs):
        organ_cases = [r for r in results if (r["gt_vol"] == cls_id).any()]
        if not organ_cases:
            continue
        organ_cases.sort(key=lambda r: r["per_class"].get(cls_id, 0), reverse=True)
        picked = [organ_cases[0]]
        if len(organ_cases) > 1:
            picked.append(organ_cases[len(organ_cases) // 2])
        picked = picked[:n_cases]

        for ci, case in enumerate(picked):
            row     = oi * n_cases + ci
            z       = best_fg_slice(case["gt_vol"], cls_id)
            img_d   = display_vol(case["img_vol"][z])
            gt_s    = case["gt_vol"][z]
            pred_s  = case["pred_vol"][z]
            img_rgb = np.stack([img_d] * 3, -1)
            dsc_val = case["per_class"].get(cls_id, 0.0)
            label   = "best" if ci == 0 else "median"

            def single_class_overlay(mask, c):
                rgba = np.zeros((*mask.shape, 4), np.float32)
                region = mask == c
                if not region.any(): return rgba
                r, g, b = hex_rgb(AMOS_HEX[c])
                border  = region & ~binary_erosion(region, iterations=1)
                rgba[region] = [r, g, b, 0.50]
                rgba[border] = [r, g, b, 1.00]
                return rgba

            for col, (base, ov) in enumerate([
                (img_rgb, None),
                (img_rgb, single_class_overlay(gt_s,   cls_id)),
                (img_rgb, single_class_overlay(pred_s, cls_id)),
                (img_rgb, error_map(gt_s == cls_id, pred_s == cls_id)),
            ]):
                ax = fig.add_subplot(gs[row, col])
                ax.imshow(base, aspect="auto", interpolation="lanczos")
                if ov is not None:
                    ax.imshow(ov, aspect="auto", interpolation="none")
                ax_clean(ax,
                         title=col_titles[col] if row == 0 else "",
                         ylabel=(f"{organ_name.replace('_',' ')}\n"
                                 f"({label}) DSC={dsc_val:.3f}") if col == 0 else "")
                if col == 2:
                    ax.text(0.97, 0.97, f"{case['case_id']}",
                            transform=ax.transAxes, fontsize=9.5,
                            ha="right", va="top", color="white",
                            bbox=dict(facecolor="black", alpha=0.55,
                                      pad=1.5, boxstyle="round,pad=0.3"))
                    ax.text(0.97, 0.03, f"DSC={dsc_val:.3f}",
                            transform=ax.transAxes, fontsize=11,
                            ha="right", va="bottom", color="white",
                            fontweight="bold",
                            bbox=dict(facecolor=hex_rgb(AMOS_HEX[cls_id]),
                                      alpha=0.88, pad=2,
                                      boxstyle="round,pad=0.4"))

    handles = [
        mpatches.Patch(color=hex_rgb(AMOS_HEX[cls_id]),
                       label=f"{name.replace('_',' ')}  (ep150={ep150:.4f})")
        for name, cls_id, ep150 in best_organs
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.00),
               ncol=3, fontsize=10, framealpha=0.95,
               title="Organ", title_fontsize=11, edgecolor="0.6")
    fig.legend(handles=[
        mpatches.Patch(color=(0.05, 0.78, 0.18), label="TP"),
        mpatches.Patch(color=(0.95, 0.12, 0.12), label="FP"),
        mpatches.Patch(color=(0.12, 0.32, 0.95), label="FN"),
    ], loc="lower right", bbox_to_anchor=(0.99, 0.00),
    ncol=3, fontsize=10, framealpha=0.95, edgecolor="0.55")

    save_fig(fig, os.path.join(output_dir, "fig4b_best_organ_spotlight"))


# =========================================================================== #
#  Fig 4 — Worst-organ spotlight
# =========================================================================== #

def fig_worst_organ_spotlight(results, output_dir):
    worst_organs = [("aorta", 8), ("stomach", 7), ("left_adrenal_gland", 12)]
    n_organs, n_cases = len(worst_organs), 2

    fig = plt.figure(figsize=(TMI_2COL, TMI_2COL * 1.00))
    fig.suptitle("Challenging Organ Segmentation — Worst / Median Cases",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(n_organs * n_cases, 4, figure=fig,
                           wspace=0.015, hspace=0.04,
                           top=0.88, bottom=0.07, left=0.13, right=0.97)
    col_titles = ["MRI Input", "Ground Truth", "Prediction", "Error Map"]

    for oi, (organ_name, cls_id) in enumerate(worst_organs):
        organ_cases = [r for r in results if (r["gt_vol"] == cls_id).any()]
        if not organ_cases: continue
        organ_cases.sort(key=lambda r: r["per_class"].get(cls_id, 0))
        picked = [organ_cases[0]]
        if len(organ_cases) > 1:
            picked.append(organ_cases[len(organ_cases) // 2])

        for ci, case in enumerate(picked[:n_cases]):
            row     = oi * n_cases + ci
            z       = best_fg_slice(case["gt_vol"], cls_id)
            img_d   = display_vol(case["img_vol"][z])
            gt_s    = case["gt_vol"][z]
            pred_s  = case["pred_vol"][z]
            img_rgb = np.stack([img_d] * 3, -1)
            dsc_val = case["per_class"].get(cls_id, 0)
            label   = "worst" if ci == 0 else "median"

            for col, (base, ov) in enumerate([
                (img_rgb, None),
                (img_rgb, organ_overlay(gt_s)),
                (img_rgb, organ_overlay(pred_s)),
                (img_rgb, error_map(gt_s, pred_s)),
            ]):
                ax = fig.add_subplot(gs[row, col])
                ax.imshow(base, aspect="auto", interpolation="lanczos")
                if ov is not None:
                    ax.imshow(ov, aspect="auto", interpolation="none")
                ax_clean(ax,
                         title=col_titles[col] if row == 0 else "",
                         ylabel=(f"{organ_name}\n({label})\nDSC={dsc_val:.3f}") if col == 0 else "")
                if col == 2:
                    ax.text(0.97, 0.03,
                            f"{case['case_id']}\nDSC={dsc_val:.3f}",
                            transform=ax.transAxes, fontsize=10,
                            ha="right", va="bottom", color="white",
                            bbox=dict(facecolor="black", alpha=0.65,
                                      pad=2, boxstyle="round,pad=0.3"))

    fig.legend(handles=[
        mpatches.Patch(color=(0.05, 0.78, 0.18), label="True Positive"),
        mpatches.Patch(color=(0.95, 0.12, 0.12), label="False Positive"),
        mpatches.Patch(color=(0.12, 0.32, 0.95), label="False Negative"),
    ], loc="lower center", bbox_to_anchor=(0.5, 0.00),
    ncol=3, fontsize=10, framealpha=0.95, edgecolor="0.55")
    save_fig(fig, os.path.join(output_dir, "fig4_worst_organ_spotlight"))


# =========================================================================== #
#  Fig 5 — Case quality grid  FIXED: cleaner column labels, no overlap
# =========================================================================== #

def fig_case_quality_grid(results, output_dir):
    if len(results) < 3:
        return

    sorted_r = sorted(results, key=lambda r: r["mean_dsc"])
    picked   = [sorted_r[-1], sorted_r[len(sorted_r)//2], sorted_r[0]]

    # FIXED: descriptive labels instead of case_id in title
    col_labels = ["Best Case", "Median Case", "Worst Case"]
    row_titles = ["Input", "GT", "Pred", "Error", "GT+Pred"]

    fig_w = TMI_2COL
    panel = fig_w / 3.2
    fig_h = panel * 5 + 1.4
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle("Qualitative Segmentation — Best / Median / Worst Cases",
                 fontsize=13, fontweight="bold", y=0.97)
    gs = gridspec.GridSpec(5, 3, figure=fig,
                           wspace=0.02, hspace=0.03,
                           top=0.88, bottom=0.14,
                           left=0.09, right=0.99)

    for col, (case, col_lbl) in enumerate(zip(picked, col_labels)):
        z       = best_fg_slice(case["gt_vol"])
        img_d   = display_vol(case["img_vol"][z])
        gt_s    = case["gt_vol"][z]
        pred_s  = case["pred_vol"][z]
        img_rgb = np.stack([img_d] * 3, -1)

        gt_ov   = organ_overlay(gt_s)
        pred_ov = organ_overlay(pred_s)
        gt_cont = np.zeros((*gt_s.shape, 4), np.float32)
        for c in range(1, len(AMOS_HEX)):
            region = gt_s == c
            if not region.any(): continue
            r2, g2, b2 = hex_rgb(AMOS_HEX[c])
            border = region & ~binary_erosion(region, iterations=2)
            gt_cont[border] = [r2, g2, b2, 1.00]

        rows_data = [
            (img_rgb, None),
            (img_rgb, gt_ov),
            (img_rgb, pred_ov),
            (img_rgb, error_map(gt_s, pred_s)),
            (img_rgb, pred_ov, gt_cont),
        ]

        for row, panels in enumerate(rows_data):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(panels[0], aspect="auto", interpolation="lanczos")
            for ov in panels[1:]:
                if ov is not None:
                    ax.imshow(ov, aspect="auto", interpolation="none")
            ax_clean(ax,
                     title=col_lbl if row == 0 else "",
                     ylabel=row_titles[row] if col == 0 else "")
            # FIXED: case_id badge on bottom of first row only
            if row == 1 and col == 0:
                ax.text(0.50, 0.03,
                        f"{case['case_id']}  DSC={case['mean_dsc']:.3f}",
                        transform=ax.transAxes, fontsize=9,
                        ha="center", va="bottom", color="white",
                        fontweight="bold",
                        bbox=dict(facecolor="#000000", alpha=0.65,
                                  pad=2, boxstyle="round,pad=0.3"))

    present_cls = set()
    for case in picked:
        z = best_fg_slice(case["gt_vol"])
        present_cls.update(int(c) for c in np.unique(case["gt_vol"][z]) if c > 0)

    organ_handles = [
        mpatches.Patch(
            color=hex_rgb(AMOS_HEX[c]),
            label=AMOS_CLASSES[c].replace("_", " ")
                  + f" ({FINAL_DICE.get(AMOS_CLASSES[c], 0):.2f})",
        )
        for c in sorted(present_cls)
        if c < len(AMOS_HEX) and FINAL_DICE.get(AMOS_CLASSES[c], 0) > 0.6
    ]
    if organ_handles:
        fig.legend(handles=organ_handles,
                   loc="lower left", bbox_to_anchor=(0.01, 0.01),
                   ncol=3, fontsize=9.5, framealpha=0.95,
                   title="Organs with DSC > 0.6", title_fontsize=11,
                   edgecolor="0.55")

    fig.legend(handles=[
        mpatches.Patch(color=(0.05, 0.78, 0.18), label="True Positive"),
        mpatches.Patch(color=(0.95, 0.12, 0.12), label="False Positive"),
        mpatches.Patch(color=(0.12, 0.32, 0.95), label="False Negative"),
    ], loc="lower right", bbox_to_anchor=(0.99, 0.01),
    ncol=1, fontsize=9.5, framealpha=0.95, edgecolor="0.55")

    save_fig(fig, os.path.join(output_dir, "fig5_case_quality_grid"))


# =========================================================================== #
#  Summary CSV
# =========================================================================== #

def write_csv(results, output_dir):
    path   = os.path.join(output_dir, "per_case_dice.csv")
    organs = AMOS_CLASSES[1:]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "mean_dsc"] + organs)
        w.writeheader()
        for r in results:
            row = {"case_id": r["case_id"], "mean_dsc": f"{r['mean_dsc']:.4f}"}
            for i, n in enumerate(organs, 1):
                row[n] = f"{r['per_class'].get(i, 0):.4f}"
            w.writerow(row)
    print("  OK  per_case_dice.csv")


# =========================================================================== #
#  CLI + main
# =========================================================================== #

def parse_args():
    BASE = "/workdir1.8t/fei27/CGT/peft_umamba/peft_umamba_Amos"
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint",
                   default="/workdir1.8t/fei27/CGT/peft_umamba/peft_umamba_Amos/outputs/peft_umamba_dataset702_mr1/best_model.pth")
    p.add_argument("--data_root",    default=f"{BASE}/data/Dataset702_AbdomenMR")
    p.add_argument("--log_path",     default=f"{BASE}/outputpeft_umamba_Amos/peft_umamba_dataset702_mr/finetune.log")
    p.add_argument("--output_dir",   default="/workdir1.8t/fei27/CGT/peft_umamba/peft_umamba_2/visualization_zoom1")
    p.add_argument("--dsc_threshold",type=float, default=0.60)
    p.add_argument("--n_slices",     type=int,   default=3)
    p.add_argument("--device",       default="cuda")
    p.add_argument("--no_model",     action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "cases"), exist_ok=True)

    print("=" * 62)
    print(" PEFT-UMamba Visualization  --  AMOS22 16-class MRI")
    print(f"  output_dir    : {args.output_dir}")
    print("=" * 62)

    print("\n[1/5]  Fig 1 -- per-class Dice bar chart ...")
    fig_per_class_dice(args.output_dir, args.dsc_threshold)

    print("\n[2/5]  Fig 3 -- learning curve ...")
    fig_learning_curve(args.log_path, args.output_dir)

    if args.no_model:
        print("\n[--no_model]  Done.\n")
        return

    if not HAS_TORCH:
        print("\n[!] PyTorch not available.\n")
        return

    if not Path(args.checkpoint).exists():
        print(f"\n[!] Checkpoint not found: {args.checkpoint}\n")
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"\n[3/5]  Inference (device={device}) ...")
    results = run_inference(args.checkpoint, args.data_root, device)
    write_csv(results, args.output_dir)

    mean_dscs = [r["mean_dsc"] for r in results]
    print(f"\n  Overall mean  : {np.mean(mean_dscs):.4f} +/- {np.std(mean_dscs):.4f}")

    runtime_dice = {}
    for c in range(1, 16):
        vals = [r["per_class"][c] for r in results if (r["gt_vol"] == c).any()]
        if vals:
            runtime_dice[AMOS_CLASSES[c]] = float(np.mean(vals))
    if runtime_dice:
        print("\n  Updating Fig 1 with inference dice values ...")
        fig_per_class_dice(args.output_dir, args.dsc_threshold, runtime_dice)

    print(f"\n[4/5]  Fig 2 -- case overlay ...")
    target_case = [r for r in results if r["case_id"] == "case_amos_0589"]
    if not target_case:
        target_case = [r for r in results if r["mean_dsc"] >= args.dsc_threshold]
    for r in target_case:
        fig_case_overlay(r, args.output_dir, n_slices=args.n_slices)

    print("\n[5/6]  Fig 4 -- worst-organ spotlight ...")
    fig_worst_organ_spotlight(results, args.output_dir)
    print("       Fig 4b -- best-organ spotlight ...")
    fig_best_organ_spotlight(results, args.output_dir)
    print("       Fig 5 -- case quality grid ...")
    fig_case_quality_grid(results, args.output_dir)

    total_figs = len(list(Path(args.output_dir).rglob("*.pdf")))
    print(f"\n{'='*62}")
    print(f"  Done.  {total_figs} PDF figures in {args.output_dir}/")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
"""
data/dataset.py — data loading for PEFT-UMamba (manuscript §4.1–4.2).

Supported layouts
─────────────────
  nnU-Net raw  (Dataset701/702/703/704):
      <root>/dataset.json, <root>/imagesTr/<case>_0000<ext>, <root>/labelsTr/<case><ext>
      NIfTI volumes (.nii.gz) → "volume" datasets; PNG/TIF/JPG → "image2d".
  Kvasir-SEG:
      <root>/images/*.jpg + <root>/masks/*.jpg   (also <root>/Kvasir-SEG/...)

Volumes (AMOS22 MRI, AbdomenCT)
  • reoriented to RAS, resampled to 1.5 mm isotropic, intensity-normalised,
    cached once as .npz under <root>/_cache_peft_umamba/
  • processed as 2-D axial slices resized to img_size × img_size
  • TRAINING batches are ordered runs of T adjacent slices from ONE volume
    (T = batch size) so SCAM sees a real slice sequence; every slice of a
    run receives the same augmentation, so the order stays meaningful
  • VALIDATION predicts whole volumes in ordered chunks of T slices

2-D images (Kvasir-SEG, EndoVis17, NeurIPS CellSeg)
  • resized to img_size, unordered batches (SCAM sees length-1 sequences)

Splits
  One fixed train/val split per dataset (split_seed, val_fraction), written to
  <root>/splits_peft_umamba.json and reused by every method and every seed.
"""

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler

from configs.config import resolve_dataset_key

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


# =========================================================================== #
#  Discovery, labels and splits
# =========================================================================== #

def _read_dataset_json(root: Path) -> Optional[dict]:
    p = root / "dataset.json"
    return json.loads(p.read_text()) if p.exists() else None


def class_names_from_json(dj: Optional[dict]) -> Optional[List[str]]:
    """Supports nnU-Net v2 {"name": idx} and v1 {"idx": "name"}."""
    if not dj or "labels" not in dj:
        return None
    labels = dj["labels"]
    pairs = []
    for k, v in labels.items():
        if isinstance(v, int):                     # v2: name -> index
            pairs.append((v, k))
        elif str(k).isdigit():                     # v1: index -> name
            pairs.append((int(k), str(v)))
    if not pairs:
        return None
    n = max(i for i, _ in pairs) + 1
    names = [f"class_{i}" for i in range(n)]
    for i, name in pairs:
        names[i] = name
    return names


def _strip_ext(name: str) -> str:
    for ext in (".nii.gz", ".nii") + IMG_EXTS:
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return Path(name).stem


def discover_test_cases(root: Path) -> List[Tuple[str, List[Path], Path]]:
    """Official labelled test set (nnU-Net raw imagesTs/ + labelsTs/), if present."""
    if not ((root / "imagesTs").is_dir() and (root / "labelsTs").is_dir()):
        return []
    try:
        return discover_cases(root, "imagesTs", "labelsTs")
    except FileNotFoundError:
        return []


def discover_cases(root: Path, img_sub: str = "imagesTr", lbl_sub: str = "labelsTr"
                   ) -> List[Tuple[str, List[Path], Path]]:
    """Returns [(case_id, [image channel files], label file), ...]."""
    dj = _read_dataset_json(root)
    ending = dj.get("file_ending") if dj else None
    img_dir, lbl_dir = root / img_sub, root / lbl_sub
    cases = []
    if img_dir.is_dir() and lbl_dir.is_dir():
        for lbl in sorted(p for p in lbl_dir.iterdir() if p.is_file()):
            if ending and not lbl.name.endswith(ending):
                continue
            case = lbl.name[: -len(ending)] if ending else _strip_ext(lbl.name)
            ext = ending or lbl.name[len(case):]
            chans = sorted(img_dir.glob(f"{case}_0[0-9][0-9][0-9]{ext}"))
            if not chans and (img_dir / f"{case}{ext}").exists():
                chans = [img_dir / f"{case}{ext}"]
            if chans:
                cases.append((case, chans, lbl))
        if cases or img_sub != "imagesTr":
            return cases
    for cand in (root, root / "Kvasir-SEG", root / "kvasir" / "Kvasir-SEG"):
        im, mk = cand / "images", cand / "masks"
        if im.is_dir() and mk.is_dir():
            masks = {_strip_ext(p.name): p for p in mk.iterdir()
                     if p.suffix.lower() in IMG_EXTS}
            for p in sorted(im.iterdir()):
                c = _strip_ext(p.name)
                if p.suffix.lower() in IMG_EXTS and c in masks:
                    cases.append((c, [p], masks[c]))
            if cases:
                return cases
    raise FileNotFoundError(
        f"No cases found under {root}. Expected nnU-Net raw "
        f"(imagesTr/ + labelsTr/) or Kvasir-SEG (images/ + masks/).")


def get_split(root: Path, case_ids: List[str], val_fraction: float,
              seed: int) -> Tuple[List[str], List[str]]:
    """Fixed split, saved to disk so every method/seed uses the same cases.
    A predefined split in <root>/splits_fixed.json ({"train": [...], "val": [...]},
    e.g. the official ACDC patient split) takes priority."""
    fixed = root / "splits_fixed.json"
    if fixed.exists():
        sp = json.loads(fixed.read_text())
        known = set(case_ids)
        tr = [c for c in sp["train"] if c in known]
        va = [c for c in sp["val"] if c in known]
        print(f"[Split] using predefined split {fixed.name}")
        return tr, va
    path = root / "splits_peft_umamba.json"
    key = f"seed{seed}_val{val_fraction}"
    store = json.loads(path.read_text()) if path.exists() else {}
    if key in store and set(store[key]["train"] + store[key]["val"]) == set(case_ids):
        return store[key]["train"], store[key]["val"]
    ids = sorted(case_ids)
    random.Random(seed).shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_fraction)))
    split = {"train": sorted(ids[n_val:]), "val": sorted(ids[:n_val])}
    store[key] = split
    try:
        path.write_text(json.dumps(store, indent=1))
    except OSError:
        print(f"[Split] could not write {path} (read-only); split kept in memory")
    return split["train"], split["val"]


# =========================================================================== #
#  Volume preprocessing (resample 1.5 mm, normalise, cache)
# =========================================================================== #

def _normalise(img: np.ndarray, modality: str) -> np.ndarray:
    if modality == "ct":
        img = np.clip(img, -125.0, 275.0)
        return ((img - img.mean()) / (img.std() + 1e-8)).astype(np.float32)
    fg = img[img > 0] if np.any(img > 0) else img.ravel()
    lo, hi = np.percentile(fg, [0.5, 99.5])
    img = np.clip(img, lo, hi)
    fg = img[img > lo] if np.any(img > lo) else img.ravel()
    return ((img - fg.mean()) / (fg.std() + 1e-8)).astype(np.float32)


def load_volume(case: str, img_path: Path, lbl_path: Path, spacing: float,
                modality: str, cache_dir: Path, resample_z: bool = True
                ) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """Returns (image [D,H,W] float32, label [D,H,W] uint8, spacing (z,y,x) mm).
    resample_z=False keeps the native slice spacing (thick-slice MRI, e.g. ACDC)
    and resamples only in-plane."""
    tag = "" if resample_z else "_xy"
    cache = cache_dir / f"{case}_{spacing:.2f}{tag}.npz"
    if cache.exists():
        z = np.load(cache)
        sp = tuple(float(v) for v in z["spacing"]) if "spacing" in z else (spacing,) * 3
        return z["image"].astype(np.float32), z["label"], sp
    import nibabel as nib
    from scipy import ndimage

    img_nii = nib.as_closest_canonical(nib.load(str(img_path)))
    lbl_nii = nib.as_closest_canonical(nib.load(str(lbl_path)))
    img = img_nii.get_fdata(dtype=np.float32)
    lbl = np.rint(lbl_nii.get_fdata(dtype=np.float32)).astype(np.uint8)
    zooms = [float(z) for z in img_nii.header.get_zooms()[:3]]
    factors = [zooms[0] / spacing, zooms[1] / spacing,
               zooms[2] / spacing if resample_z else 1.0]
    img = ndimage.zoom(img, factors, order=1)
    lbl = ndimage.zoom(lbl, factors, order=0)
    img = _normalise(img.transpose(2, 1, 0), modality)       # [X,Y,Z] → [Z,Y,X]
    lbl = lbl.transpose(2, 1, 0).astype(np.uint8)
    sp = (spacing if resample_z else zooms[2], spacing, spacing)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, image=img.astype(np.float16), label=lbl,
                        spacing=np.array(sp, dtype=np.float32))
    return img, lbl, sp


# =========================================================================== #
#  Tensor helpers and augmentation
# =========================================================================== #

def resize_image(x: torch.Tensor, size: int) -> torch.Tensor:          # [N,C,H,W]
    if x.shape[-2:] == (size, size):
        return x
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


def resize_label(y: torch.Tensor, size: int) -> torch.Tensor:          # [N,H,W]
    if y.shape[-2:] == (size, size):
        return y
    return F.interpolate(y[:, None].float(), size=(size, size),
                         mode="nearest")[:, 0].long()


def augment(img: torch.Tensor, lbl: torch.Tensor, rgb: bool,
            rotation_deg: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same random transform for every element of the leading dimension
    (so an ordered slice run stays geometrically consistent).
    img [N,C,H,W] (RGB in [0,1] before normalisation), lbl [N,H,W]."""
    if torch.rand(1).item() < 0.5:
        img, lbl = img.flip(-1), lbl.flip(-1)
    if torch.rand(1).item() < 0.5:
        img, lbl = img.flip(-2), lbl.flip(-2)
    if rotation_deg > 0:
        a = math.radians((torch.rand(1).item() * 2 - 1) * rotation_deg)
        theta = torch.tensor([[math.cos(a), -math.sin(a), 0.0],
                              [math.sin(a), math.cos(a), 0.0]],
                             dtype=img.dtype).expand(img.shape[0], 2, 3)
        grid = F.affine_grid(theta, list(img.shape), align_corners=False)
        img = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros",
                            align_corners=False)
        lbl = F.grid_sample(lbl[:, None].float(), grid, mode="nearest",
                            padding_mode="zeros", align_corners=False)[:, 0].long()
    if rgb:                                                  # colour jitter
        b = 0.8 + 0.4 * torch.rand(1).item()
        c = 0.8 + 0.4 * torch.rand(1).item()
        mean = img.mean(dim=(-3, -2, -1), keepdim=True)
        img = ((img * b - mean) * c + mean).clamp(0, 1)
    return img, lbl


def normalise_rgb(img: torch.Tensor) -> torch.Tensor:          # [...,3,H,W] in [0,1]
    return (img - IMAGENET_MEAN.to(img)) / IMAGENET_STD.to(img)


def slices_to_input(vol: np.ndarray, size: int) -> torch.Tensor:
    """[T,H,W] normalised grey slices → [T,3,size,size] model input."""
    x = torch.from_numpy(np.ascontiguousarray(vol, dtype=np.float32))[:, None]
    return resize_image(x, size).repeat(1, 3, 1, 1)


# =========================================================================== #
#  Datasets
# =========================================================================== #

class OrderedSliceRuns(Dataset):
    """Training item = run of T consecutive axial slices from one volume.
    Use with DataLoader(batch_size=None): one item is one batch."""

    def __init__(self, volumes: List[Tuple[np.ndarray, np.ndarray]], run_len: int,
                 img_size: int, rotation_deg: float):
        self.vols = volumes
        self.T = run_len
        self.size = img_size
        self.rot = rotation_deg
        runs = [max(1, math.ceil(v[0].shape[0] / run_len)) for v in volumes]
        self.cum = np.cumsum(runs)

    def __len__(self):
        return int(self.cum[-1])

    def __getitem__(self, i):
        v = int(np.searchsorted(self.cum, i, side="right"))
        img, lbl = self.vols[v][0], self.vols[v][1]
        D = img.shape[0]
        T = min(self.T, D)                    # short stacks: whole volume, no padding
        start = int(torch.randint(0, D - T + 1, (1,)))
        idx = np.arange(start, start + T)
        x = slices_to_input(img[idx], self.size)
        y = resize_label(torch.from_numpy(lbl[idx].astype(np.int64)), self.size)
        x, y = augment(x, y, rgb=False, rotation_deg=self.rot)
        return {"image": x.contiguous(), "label": y.contiguous()}


class VolumeVal(Dataset):
    def __init__(self, volumes, case_ids):
        self.vols, self.cases = volumes, case_ids

    def __len__(self):
        return len(self.vols)

    def __getitem__(self, i):
        img, lbl, sp = self.vols[i]
        return {"image": img, "label": lbl, "spacing": np.array(sp), "case": self.cases[i]}


def _read_array(path: Path) -> np.ndarray:
    """PNG/JPG/TIF via PIL, 2-D NIfTI (nnU-Net natural-image layout) via nibabel."""
    name = path.name.lower()
    if name.endswith(".nii") or name.endswith(".nii.gz"):
        import nibabel as nib
        return np.squeeze(np.asarray(nib.load(str(path)).dataobj))
    from PIL import Image
    return np.asarray(Image.open(path))


def _load_image_file(paths: List[Path]) -> np.ndarray:
    """Returns [H,W,3] float32 in [0,1]."""
    if len(paths) == 1:
        arr = _read_array(paths[0]).astype(np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, -1)
        elif arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            arr = arr.transpose(1, 2, 0)
        arr = arr[..., :3]
    else:
        chans = [_read_array(p).astype(np.float32) for p in paths[:3]]
        while len(chans) < 3:
            chans.append(chans[-1])
        arr = np.stack(chans, -1)
    return arr / (255.0 if arr.max() > 1.0 else 1.0)


def _load_label_file(path: Path, binary: bool) -> np.ndarray:
    arr = _read_array(path)
    if arr.ndim == 3:
        arr = arr[..., 0] if arr.shape[-1] in (3, 4) else arr[0]
    if binary:
        return (arr > 127).astype(np.int64) if arr.max() > 1 else (arr > 0).astype(np.int64)
    return np.rint(arr).astype(np.int64)


class Image2DDataset(Dataset):
    def __init__(self, cases, img_size: int, train: bool, binary_masks: bool,
                 rotation_deg: float):
        self.cases = cases
        self.size = img_size
        self.train = train
        self.binary = binary_masks
        self.rot = rotation_deg

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        case, img_paths, lbl_path = self.cases[i]
        img = torch.from_numpy(_load_image_file(img_paths)).permute(2, 0, 1)[None]
        lbl = torch.from_numpy(_load_label_file(lbl_path, self.binary))[None]
        img, lbl = resize_image(img, self.size), resize_label(lbl, self.size)
        if self.train:
            img, lbl = augment(img, lbl, rgb=True, rotation_deg=self.rot)
        return {"image": normalise_rgb(img[0]).contiguous(),
                "label": lbl[0].contiguous(), "case": case}


def _read_instance_mask(path: Path) -> np.ndarray:
    """Instance-id mask (16/32-bit TIFF or PNG)."""
    try:
        import tifffile
        return np.asarray(tifffile.imread(str(path))).squeeze().astype(np.int64)
    except Exception:
        return _read_array(path).squeeze().astype(np.int64)


def instances_to_semantic(inst: np.ndarray) -> np.ndarray:
    """0 background, 1 cell interior, 2 cell boundary (4-neighbourhood)."""
    b = np.zeros(inst.shape, bool)
    b[1:, :] |= inst[1:, :] != inst[:-1, :]
    b[:-1, :] |= inst[:-1, :] != inst[1:, :]
    b[:, 1:] |= inst[:, 1:] != inst[:, :-1]
    b[:, :-1] |= inst[:, :-1] != inst[:, 1:]
    sem = np.zeros(inst.shape, np.int64)
    sem[inst > 0] = 1
    sem[(inst > 0) & b] = 2
    return sem


def discover_cellseg_val(root: Path):
    """Official NeurIPS-CellSeg tuning set in the U-Mamba layout:
    imagesVal/<case>_0000.<ext> + labelsVal-instance-mask/<case>_label.tiff"""
    im_dir, lb_dir = root / "imagesVal", root / "labelsVal-instance-mask"
    if not (im_dir.is_dir() and lb_dir.is_dir()):
        return []
    labels = {}
    for p in lb_dir.iterdir():
        c = _strip_ext(p.name)
        labels[c[:-6] if c.endswith("_label") else c] = p
    out = []
    for p in sorted(im_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        c = _strip_ext(p.name)
        c = c[:-5] if c.endswith("_0000") else c
        if c in labels:
            out.append((c, [p], labels[c]))
    return out


class CellValDataset(Dataset):
    """Official CellSeg evaluation: image resized to the network input; label
    returned (i) as 3-class semantic map at the input size, for DSC, and
    (ii) as the native-resolution instance mask, for instance F1."""

    def __init__(self, cases, img_size: int):
        self.cases, self.size = cases, img_size

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        case, img_paths, lbl_path = self.cases[i]
        img = torch.from_numpy(_load_image_file(img_paths)).permute(2, 0, 1)[None]
        inst = _read_instance_mask(lbl_path)
        sem = torch.from_numpy(instances_to_semantic(inst))[None]
        return {"image": normalise_rgb(resize_image(img, self.size)[0]).contiguous(),
                "label": resize_label(sem, self.size)[0].contiguous(),
                "instances": torch.from_numpy(inst), "case": case}


# =========================================================================== #
#  Builder
# =========================================================================== #

def build_loaders(cfg):
    """Builds loaders, fixes num_classes / class names from dataset.json and
    returns (train_loader, val_loader, class_names)."""
    dc, tc = cfg.data, cfg.train
    key = resolve_dataset_key(dc.dataset)
    root = Path(dc.data_root)
    cases = discover_cases(root)
    dj = _read_dataset_json(root)
    names = class_names_from_json(dj)
    if key == "kvasir" and names is None:
        names = ["background", "polyp"]
    if names is None:
        names = [f"class_{i}" for i in range(dc.num_classes)]
    dc.num_classes = len(names)
    cfg.model.num_classes = len(names)

    by_id = {c[0]: c for c in cases}
    test = [] if (root / "splits_fixed.json").exists() else discover_test_cases(root)
    if test:
        # official labelled test set: train on all of imagesTr, report on imagesTs
        # (the same test cases as the published baselines)
        tr_ids, va_ids = sorted(by_id), [c[0] for c in test]
        by_id.update({c[0]: c for c in test})
        print(f"[Split] official test set imagesTs/labelsTs ({len(va_ids)} cases)")
    else:
        tr_ids, va_ids = get_split(root, list(by_id), dc.val_fraction, dc.split_seed)
    print(f"[Data] {key}: {len(by_id)} cases → {len(tr_ids)} train / "
          f"{len(va_ids)} val, {len(names)} classes, kind={dc.kind}")

    if dc.kind == "volume":
        cache = root / "_cache_peft_umamba"

        def load(ids):
            out = []
            for n, cid in enumerate(ids, 1):
                c, imgs, lbl = by_id[cid]
                out.append(load_volume(c, imgs[0], lbl, dc.spacing, dc.modality, cache,
                                       getattr(dc, "resample_z", True)))
                if n % 10 == 0 or n == len(ids):
                    print(f"  loaded {n}/{len(ids)} volumes")
            return out

        train_ds = OrderedSliceRuns(load(tr_ids), tc.batch_size, dc.img_size,
                                    tc.rotation_deg)
        val_ds = VolumeVal(load(va_ids), va_ids)
        ipe = getattr(tc, "iters_per_epoch", None)
        sampler = (RandomSampler(train_ds, replacement=True, num_samples=ipe)
                   if ipe else RandomSampler(train_ds))
        train_loader = DataLoader(train_ds, batch_size=None, sampler=sampler,
                                  num_workers=dc.num_workers, pin_memory=True,
                                  persistent_workers=dc.num_workers > 0)
        val_loader = DataLoader(val_ds, batch_size=None, shuffle=False,
                                num_workers=0)
    else:
        binary = key == "kvasir"
        cell_val = discover_cellseg_val(root) if key == "dataset703" else []
        if cell_val:
            # official tuning set: train on all of imagesTr, evaluate on imagesVal
            tr_ids = sorted(c[0] for c in cases)
            print(f"[Split] official CellSeg tuning set imagesVal ({len(cell_val)} images, "
                  f"instance masks); training on all {len(tr_ids)} imagesTr")
        train_ds = Image2DDataset([by_id[i] for i in tr_ids], dc.img_size, True,
                                  binary, tc.rotation_deg)
        if cell_val:
            val_ds = CellValDataset(cell_val, dc.img_size)
            val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
        else:
            val_ds = Image2DDataset([by_id[i] for i in va_ids], dc.img_size, False,
                                    binary, 0.0)
        train_loader = DataLoader(train_ds, batch_size=tc.batch_size, shuffle=True,
                                  num_workers=dc.num_workers, pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=dc.num_workers > 0)
        if not cell_val:
            val_loader = DataLoader(val_ds, batch_size=tc.batch_size, shuffle=False,
                                    num_workers=dc.num_workers)
    return train_loader, val_loader, names


# =========================================================================== #
#  Volume inference (ordered chunks → SCAM sees real slice sequences)
# =========================================================================== #

AMP_DTYPE = torch.float16          # train.py switches to bfloat16 for full fine-tuning


def set_amp_dtype(dtype):
    global AMP_DTYPE
    AMP_DTYPE = dtype


@torch.no_grad()
def predict_volume(model, vol: np.ndarray, img_size: int, chunk: int,
                   device, use_amp: bool = True) -> np.ndarray:
    D, H, W = vol.shape
    pred = np.zeros((D, H, W), dtype=np.uint8)
    for s in range(0, D, chunk):
        x = slices_to_input(vol[s:s + chunk], img_size).to(device)
        with torch.autocast(device_type=device.type, dtype=AMP_DTYPE,
                            enabled=use_amp and device.type == "cuda"):
            logits = model(x)
        logits = F.interpolate(logits.float(), size=(H, W), mode="bilinear",
                               align_corners=False)
        pred[s:s + chunk] = logits.argmax(1).cpu().numpy().astype(np.uint8)
    return pred

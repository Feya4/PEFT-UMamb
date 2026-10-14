"""
summarize.py — mean ± std over seeds for every experiment in summary.csv.

    python summarize.py                      # outputs/summary.csv
    python summarize.py --csv path/summary.csv --dataset dataset702
"""
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=str(Path(__file__).resolve().parent / "outputs" / "summary.csv"))
ap.add_argument("--dataset")
args = ap.parse_args()

rows = list(csv.DictReader(open(args.csv)))
groups = defaultdict(list)
for r in rows:
    if args.dataset and r["dataset"] != args.dataset:
        continue
    groups[re.sub(r"_s\d+$", "", r["exp"])].append(r)


def ms(vals):
    vals = [float(v) for v in vals if v not in ("", "nan") and not math.isnan(float(v))]
    if not vals:
        return "—"
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
    return f"{m:.4f} ± {sd:.4f}"


hdr = f"{'experiment':42s} {'n':>2s} {'adapt M':>8s} {'train M':>8s} {'mem GB':>7s} " \
      f"{'ms':>6s}  {'Dice (seeds)':>18s}  {'HD95':>18s}  {'inst F1':>18s}"
print(hdr)
print("-" * len(hdr))
for exp in sorted(groups):
    g = groups[exp]
    print(f"{exp:42s} {len(g):2d} {g[0]['adapt_M']:>8s} {g[0]['trainable_M']:>8s} "
          f"{g[0]['train_mem_GB']:>7s} {g[0]['infer_ms']:>6s}  "
          f"{ms(r['dice'] for r in g):>18s}  {ms(r['hd95'] for r in g):>18s}  "
          f"{ms(r['instance_f1'] for r in g):>18s}")

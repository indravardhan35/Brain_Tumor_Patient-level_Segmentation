import os
import csv
from collections import defaultdict

import cv2
import numpy as np
import torch

from dataset import BrainTumorDataset
from model import LightweightUNet

CHECKPOINT = "best_model.pth"   # baseline model; change to analyse another
OUT_DIR = "error_analysis"
GLIOMA = 2
N_WORST = 10

os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = BrainTumorDataset(split_file="patient_level_split.json",
                            split_name="test", augment=False)
print("Record keys:", list(dataset.records[0].keys()))  # check patient-ID key name

model = LightweightUNet(num_classes=1, pretrained=False).to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()


def patient_of(record):
    for k in ("pid", "PID", "patient_id", "patient"):
        if k in record:
            return str(record[k])
    return "unknown"


rows = []
cache = {}  # idx -> (image, mask, pred) for plotting worst cases

with torch.no_grad():
    for idx in range(len(dataset)):
        image, mask, label = dataset[idx]
        if label != GLIOMA:
            continue
        out = model(image.unsqueeze(0).to(device))
        pred = (out > 0.5).float().cpu()[0, 0].numpy()
        gt = mask[0].numpy()

        tp = (pred * gt).sum()
        dice = 2 * tp / (pred.sum() + gt.sum() + 1e-8)
        precision = tp / (pred.sum() + 1e-8)
        recall = tp / (gt.sum() + 1e-8)

        rows.append({
            "idx": idx,
            "patient": patient_of(dataset.records[idx]),
            "gt_area": int(gt.sum()),
            "pred_area": int(pred.sum()),
            "dice": float(dice),
            "precision": float(precision),
            "recall": float(recall),
        })
        cache[idx] = (image[0].numpy(), gt, pred)

dices = np.array([r["dice"] for r in rows])
print(f"\nGlioma slices: {len(rows)} | mean Dice = {dices.mean():.4f} "
      f"(should be close to your baseline 0.6694)")

# ---- 1. Dice by tumour size (tertiles of ground-truth area) ----
areas = np.array([r["gt_area"] for r in rows])
lo, hi = np.percentile(areas, [33.3, 66.6])
buckets = {"small": [], "medium": [], "large": []}
for r in rows:
    key = "small" if r["gt_area"] <= lo else ("medium" if r["gt_area"] <= hi else "large")
    buckets[key].append(r)

print(f"\nDice by tumour size (256x256 px area; cutoffs {lo:.0f} / {hi:.0f}):")
for name, rs in buckets.items():
    d = np.mean([r["dice"] for r in rs])
    p = np.mean([r["precision"] for r in rs])
    rc = np.mean([r["recall"] for r in rs])
    print(f"  {name:6s} n={len(rs):3d}  Dice={d:.4f}  precision={p:.3f}  recall={rc:.3f}")

# ---- 2. Dice by patient ----
by_patient = defaultdict(list)
for r in rows:
    by_patient[r["patient"]].append(r["dice"])
print("\nDice by patient (glioma):")
for pid, ds in sorted(by_patient.items(), key=lambda kv: np.mean(kv[1])):
    print(f"  patient {pid}: n={len(ds):3d}  mean Dice={np.mean(ds):.4f}")

# ---- 3. Failure summary ----
print(f"\nSlices with Dice < 0.5: {(dices < 0.5).sum()} / {len(rows)}")
print(f"Slices with Dice < 0.1 (near-total miss): {(dices < 0.1).sum()}")

# ---- 4. Save CSV + worst-case images ----
with open(os.path.join(OUT_DIR, "glioma_slice_metrics.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

for r in sorted(rows, key=lambda r: r["dice"])[:N_WORST]:
    img, gt, pred = cache[r["idx"]]
    base = (img * 255).astype(np.uint8)
    panels = [
        cv2.cvtColor(base, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor((gt * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
        cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
    ]
    # overlay panel: green = ground truth, red = prediction
    overlay = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    overlay[gt > 0.5] = (0.5 * overlay[gt > 0.5] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    overlay[pred > 0.5] = (0.5 * overlay[pred > 0.5] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    panels.append(overlay)
    strip = cv2.hconcat(panels)
    cv2.putText(strip, f"Dice {r['dice']:.3f} | GT {r['gt_area']}px | pred {r['pred_area']}px",
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.imwrite(os.path.join(OUT_DIR, f"worst_idx{r['idx']}_dice{r['dice']:.2f}.png"), strip)

print(f"\nSaved CSV and {N_WORST} worst-case images to ./{OUT_DIR}/")
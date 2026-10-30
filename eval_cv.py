import argparse, csv, json
import torch
from torch.utils.data import DataLoader
from dataset import BrainTumorDataset
from model import LightweightUNet
from model_aspp import ASPPUNet

ap = argparse.ArgumentParser()
ap.add_argument("--split_file", required=True)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--model", choices=["baseline", "aspp"], required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(args.split_file) as fh:
    records = json.load(fh)["test"]
ds = BrainTumorDataset(split_file=args.split_file, split_name="test", augment=False)
assert len(ds) == len(records)
loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

model = (ASPPUNet(num_classes=1, pretrained=False) if args.model == "aspp"
         else LightweightUNet(num_classes=1, pretrained=False)).to(device)
model.load_state_dict(torch.load(args.checkpoint, map_location=device))
model.eval()

rows, i = [], 0
with torch.no_grad():
    for images, masks, _ in loader:
        pred = (model(images.to(device)).cpu() > 0.5).float().flatten(1)
        gt = masks.flatten(1)
        inter = (pred * gt).sum(1)
        dice = (2 * inter + 1e-8) / (pred.sum(1) + gt.sum(1) + 1e-8)
        for d in dice.tolist():
            r = records[i]
            rows.append((r["patient_id"], r["label"], r["filename"], d))
            i += 1

with open(args.out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["patient_id", "label", "filename", "dice"])
    w.writerows(rows)

names = {1: "meningioma", 2: "glioma", 3: "pituitary"}
for lab, nm in names.items():
    v = [d for _, l, _, d in rows if l == lab]
    if v:
        print(f"  {nm} (n={len(v)}): Mean Dice = {sum(v)/len(v):.4f}")
print(f"Overall: {sum(d for *_, d in rows)/len(rows):.4f}")
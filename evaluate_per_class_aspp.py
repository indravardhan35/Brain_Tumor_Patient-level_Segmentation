import sys
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from dataset import BrainTumorDataset
from model_aspp import ASPPUNet
from model import LightweightUNet

ckpt = sys.argv[1] if len(sys.argv) > 1 else "best_model_aspp_seed0.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = BrainTumorDataset("patient_level_split.json", "test", augment=False)
loader = DataLoader(ds, batch_size=1, shuffle=False)
model = (ASPPUNet(num_classes=1, pretrained=False) if "aspp" in ckpt
         else LightweightUNet(num_classes=1, pretrained=False)).to(device)
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()

names = {1: "meningioma", 2: "glioma", 3: "pituitary"}
per_class, overall = defaultdict(list), []
with torch.no_grad():
    for img, mask, label in loader:
        pred = (model(img.to(device)).cpu() > 0.5).float()
        inter = (pred * mask).sum()
        d = (2 * inter / (pred.sum() + mask.sum() + 1e-8)).item()
        per_class[label.item()].append(d)
        overall.append(d)

print(f"Checkpoint: {ckpt} | test size {len(ds)}")
for k in sorted(per_class):
    print(f"  {names[k]} (n={len(per_class[k])}): Mean Dice = {sum(per_class[k]) / len(per_class[k]):.4f}")
print(f"Overall Test Dice: {sum(overall) / len(overall):.4f}")
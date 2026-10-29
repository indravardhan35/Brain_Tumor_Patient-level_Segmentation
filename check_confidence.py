import pandas as pd
import torch
from dataset import BrainTumorDataset
from model import LightweightUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds = BrainTumorDataset(split_file="patient_level_split.json", split_name="test", augment=False)
model = LightweightUNet(num_classes=1, pretrained=False).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

df = pd.read_csv("error_analysis/glioma_slice_metrics.csv")
miss = df[df.dice < 0.1]

def dice_at(pred, gt, t):
    p = (pred > t).float()
    return (2 * (p * gt).sum() / (p.sum() + gt.sum() + 1e-8)).item()

print("idx  patient     gt_area  max_prob  dice@0.5  dice@0.3  dice@0.2")
with torch.no_grad():
    for _, r in miss.iterrows():
        img, mask, _ = ds[int(r.idx)]
        out = model(img.unsqueeze(0).to(device)).cpu()[0, 0]
        gt = mask[0]
        print(f"{int(r.idx):4d} {r.patient:11s} {int(r.gt_area):7d}  {out.max():.3f}    "
              f"{dice_at(out, gt, 0.5):.3f}     {dice_at(out, gt, 0.3):.3f}     {dice_at(out, gt, 0.2):.3f}")
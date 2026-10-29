import cv2
import numpy as np
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
miss = df[df.dice < 0.1].sort_values(["patient", "idx"])

tiles = []
with torch.no_grad():
    for _, r in miss.iterrows():
        img, mask, _ = ds[int(r.idx)]
        pred = (model(img.unsqueeze(0).to(device)).cpu()[0, 0] > 0.5).numpy()
        gt = mask[0].numpy() > 0.5
        base = (img[0].numpy() * 255).astype(np.uint8)
        plain = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        ov = plain.copy()
        ov[gt] = (0.5 * ov[gt] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        ov[pred] = (0.5 * ov[pred] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
        pair = cv2.hconcat([plain, ov])
        cv2.putText(pair, f"idx {int(r.idx)} {r.patient}", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        tiles.append(pair)

while len(tiles) % 3:
    tiles.append(np.zeros_like(tiles[0]))
rows = [cv2.hconcat(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
cv2.imwrite("error_analysis/miss_montage.png", cv2.vconcat(rows))
print("Saved error_analysis/miss_montage.png")
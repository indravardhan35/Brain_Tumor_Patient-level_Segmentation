import argparse
import copy
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import BrainTumorDataset
from model_aspp import ASPPUNet
from losses import CombinedLoss   # same loss as the baseline

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--patience", type=int, default=10)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds = BrainTumorDataset("patient_level_split.json", "train", augment=True)
val_ds = BrainTumorDataset("patient_level_split.json", "val", augment=False)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

model = ASPPUNet(num_classes=1, pretrained=True).to(device)
print("Parameters:", sum(p.numel() for p in model.parameters()))

criterion = CombinedLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)


def batch_dice(pred, gt, thr=0.5, eps=1e-8):
    """Per-slice Dice, averaged over the batch."""
    p = (pred > thr).float().flatten(1)
    g = gt.flatten(1)
    inter = (p * g).sum(1)
    return ((2 * inter + eps) / (p.sum(1) + g.sum(1) + eps)).mean().item()


best_val, best_state, bad_epochs = -1.0, None, 0
for epoch in range(1, args.epochs + 1):
    model.train()
    train_loss = 0.0
    for images, masks, _ in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    train_loss /= len(train_ds)

    model.eval()
    val_dice = 0.0
    with torch.no_grad():
        for images, masks, _ in val_loader:
            images, masks = images.to(device), masks.to(device)
            val_dice += batch_dice(model(images), masks) * images.size(0)
    val_dice /= len(val_ds)
    scheduler.step(val_dice)

    print(f"Epoch {epoch:3d} | train loss {train_loss:.4f} | val Dice {val_dice:.4f} "
          f"| lr {optimizer.param_groups[0]['lr']:.1e}")

    if val_dice > best_val:
        best_val, bad_epochs = val_dice, 0
        best_state = copy.deepcopy(model.state_dict())
        torch.save(best_state, f"best_model_aspp_seed{args.seed}.pth")
    else:
        bad_epochs += 1
        if bad_epochs >= args.patience:
            print("Early stopping.")
            break

print(f"\nBest val Dice (seed {args.seed}): {best_val:.4f}")
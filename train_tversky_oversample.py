import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import BrainTumorDataset
from model import LightweightUNet
from losses import CombinedTverskyLoss, dice_score
import time
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_dataset = BrainTumorDataset(split_name="train", augment=True)
val_dataset = BrainTumorDataset(split_name="val", augment=False)

# ---- Build sample weights: glioma gets sampled more often ----
with open("patient_level_split.json", "r") as f:
    split_data = json.load(f)

train_labels = [r["label"] for r in split_data["train"]]

# Give glioma (label 2) a higher weight, others weight 1.0
sample_weights = []
for label in train_labels:
    if label == 2:  # glioma
        sample_weights.append(2.5)
    else:
        sample_weights.append(1.0)

sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler, num_workers=0)  # sampler instead of shuffle
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

model = LightweightUNet(num_classes=1, pretrained=True).to(device)
criterion = CombinedTverskyLoss(tversky_weight=0.7, bce_weight=0.3, alpha=0.3, beta=0.7)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

NUM_EPOCHS = 40
PATIENCE = 7
best_val_dice = 0.0
epochs_without_improvement = 0

for epoch in range(NUM_EPOCHS):
    start_time = time.time()
    
    model.train()
    train_loss = 0.0
    for images, masks, labels in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    with torch.no_grad():
        for images, masks, labels in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            val_dice += dice_score(outputs, masks)
    
    val_loss /= len(val_loader)
    val_dice /= len(val_loader)
    
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    elapsed = time.time() - start_time
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | LR: {current_lr:.6f} | Time: {elapsed:.1f}s")
    
    if val_dice > best_val_dice:
        best_val_dice = val_dice
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "best_model_tversky_oversample.pth")
        print(f"  -> New best model saved (Val Dice: {val_dice:.4f})")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

print(f"\nTraining complete! Best Val Dice (Tversky + oversampling): {best_val_dice:.4f}")
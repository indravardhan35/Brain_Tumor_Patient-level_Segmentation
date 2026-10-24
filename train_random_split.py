import torch
from torch.utils.data import DataLoader
from dataset import BrainTumorDataset
from model import LightweightUNet
from losses import CombinedLoss, dice_score
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Use the RANDOM split this time (split_file points to random_split.json)
train_dataset = BrainTumorDataset(split_file="random_split.json", split_name="train", augment=True)
val_dataset = BrainTumorDataset(split_file="random_split.json", split_name="val", augment=False)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

model = LightweightUNet(num_classes=1, pretrained=True).to(device)
criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
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
        torch.save(model.state_dict(), "best_model_random_split.pth")  # different filename!
        print(f"  -> New best model saved (Val Dice: {val_dice:.4f})")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

print(f"\nTraining complete! Best Val Dice (random split): {best_val_dice:.4f}")
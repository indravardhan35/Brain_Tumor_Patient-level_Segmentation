import torch
from torch.utils.data import DataLoader
from dataset import BrainTumorDataset
from model import LightweightUNet
from losses import DiceLoss, dice_score
import time

# ---- Setup ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# For the quick sanity check: use a SMALL subset first
SANITY_CHECK = True
SUBSET_SIZE = 50  # just 50 images, to test the pipeline quickly

train_dataset = BrainTumorDataset(split_name="train")
val_dataset = BrainTumorDataset(split_name="val")

if SANITY_CHECK:
    from torch.utils.data import Subset
    train_dataset = Subset(train_dataset, range(min(SUBSET_SIZE, len(train_dataset))))
    val_dataset = Subset(val_dataset, range(min(20, len(val_dataset))))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

# ---- Model, loss, optimizer ----
model = LightweightUNet(num_classes=1, pretrained=True).to(device)
criterion = DiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ---- Training loop ----
NUM_EPOCHS = 3  # just a few, for the sanity check

for epoch in range(NUM_EPOCHS):
    start_time = time.time()
    
    # Training phase
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
    
    # Validation phase
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
    
    elapsed = time.time() - start_time
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Time: {elapsed:.1f}s")

print("\nSanity check complete!")
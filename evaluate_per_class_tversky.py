import torch
from torch.utils.data import DataLoader
from dataset import BrainTumorDataset
from model import LightweightUNet
from losses import dice_score
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_dataset = BrainTumorDataset(split_file="patient_level_split.json", split_name="test", augment=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = LightweightUNet(num_classes=1, pretrained=False).to(device)
model.load_state_dict(torch.load("best_model_tversky.pth", map_location=device))
model.eval()

label_names = {1: "meningioma", 2: "glioma", 3: "pituitary"}
class_dice_scores = defaultdict(list)
overall_dice_scores = []

with torch.no_grad():
    for images, masks, labels in test_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        
        dice = dice_score(outputs, masks)
        label = labels.item()
        
        class_dice_scores[label].append(dice)
        overall_dice_scores.append(dice)

print(f"Test set size: {len(test_dataset)}\n")
print("Per-class Dice scores (TVERSKY MODEL):")
for label_id in sorted(class_dice_scores.keys()):
    scores = class_dice_scores[label_id]
    avg = sum(scores) / len(scores)
    print(f"  {label_names[label_id]} (n={len(scores)}): Mean Dice = {avg:.4f}")

overall_avg = sum(overall_dice_scores) / len(overall_dice_scores)
print(f"\nOverall Test Dice: {overall_avg:.4f}")
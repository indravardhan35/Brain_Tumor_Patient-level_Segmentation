import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """Standard Dice Loss for binary segmentation"""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice_coeff = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice_coeff  # loss = 1 - dice score


def dice_score(pred, target, threshold=0.5, smooth=1e-6):
    """Compute Dice score (for evaluation, not training) """
    pred = (pred > threshold).float()
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()

class CombinedLoss(nn.Module):
    """Combines Dice Loss and Binary Cross-Entropy for better handling of class imbalance"""
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return self.dice_weight * dice + self.bce_weight * bce

class WeightedCombinedLoss(nn.Module):
    """
    Combined Dice + BCE loss, with extra weight applied to glioma samples.
    label: 1=meningioma, 2=glioma, 3=pituitary
    """
    def __init__(self, dice_weight=0.5, bce_weight=0.5, glioma_weight=2.0):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss(reduction='none')  # per-pixel, so we can weight per-sample
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.glioma_weight = glioma_weight

    def forward(self, pred, target, labels):
        # Dice loss (computed per-batch as before)
        dice = self.dice_loss(pred, target)
        
        # BCE loss computed per-sample, then weighted by class before averaging
        bce_per_pixel = self.bce_loss(pred, target)  # shape: [batch, 1, H, W]
        bce_per_sample = bce_per_pixel.view(bce_per_pixel.size(0), -1).mean(dim=1)  # [batch]
        
        # Build per-sample weights: glioma (label==2) gets extra weight
        weights = torch.ones_like(bce_per_sample)
        for i, label in enumerate(labels):
            if label.item() == 2:  # glioma
                weights[i] = self.glioma_weight
        
        weighted_bce = (bce_per_sample * weights).mean()
        
        return self.dice_weight * dice + self.bce_weight * weighted_bce
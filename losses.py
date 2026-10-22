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
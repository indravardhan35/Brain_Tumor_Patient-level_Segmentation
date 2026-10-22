import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import json
import os
import cv2
import random

class BrainTumorDataset(Dataset):
    def __init__(self, split_file="patient_level_split.json", split_name="train", image_size=256, augment=False):
        """
        augment: if True, applies random flips/rotation/brightness shifts.
                 Only use augment=True for the training set.
        """
        with open(split_file, "r") as f:
            all_splits = json.load(f)
        
        self.records = all_splits[split_name]
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def _apply_augmentation(self, image, mask):
        # Random horizontal flip
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        
        # Random vertical flip
        if random.random() > 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
        
        # Random rotation (small angles: -15 to +15 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            h, w = image.shape
            center = (w // 2, h // 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)
            mask = cv2.warpAffine(mask, rot_matrix, (w, h), flags=cv2.INTER_NEAREST)
        
        # Random brightness/contrast shift (image only, not mask)
        if random.random() > 0.5:
            brightness = random.uniform(-0.1, 0.1)
            contrast = random.uniform(0.9, 1.1)
            image = image * contrast + brightness
            image = np.clip(image, 0, 1)
        
        return image, mask

    def __getitem__(self, idx):
        record = self.records[idx]
        file_path = os.path.join(record["folder"], record["filename"])
        
        with h5py.File(file_path, 'r') as f:
            cjdata = f['cjdata']
            image = np.array(cjdata['image'], dtype=np.float32)
            mask = np.array(cjdata['tumorMask'], dtype=np.float32)
        
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        mask = (mask > 0.5).astype(np.float32)
        
        # Apply augmentation only if enabled (training set)
        if self.augment:
            image, mask = self._apply_augmentation(image, mask)
            mask = (mask > 0.5).astype(np.float32)  # re-binarize after rotation interpolation
        
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)
        
        image_tensor = torch.from_numpy(image.copy()).float()
        mask_tensor = torch.from_numpy(mask.copy()).float()
        label = record["label"]
        
        return image_tensor, mask_tensor, label
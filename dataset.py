import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import json
import os
import cv2

class BrainTumorDataset(Dataset):
    def __init__(self, split_file="patient_level_split.json", split_name="train", image_size=256):
        """
        split_file: path to patient_level_split.json
        split_name: "train", "val", or "test"
        image_size: resize target (e.g., 256x256), since 512x512 is large for a lightweight model
        """
        with open(split_file, "r") as f:
            all_splits = json.load(f)
        
        self.records = all_splits[split_name]
        self.image_size = image_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        file_path = os.path.join(record["folder"], record["filename"])
        
        with h5py.File(file_path, 'r') as f:
            cjdata = f['cjdata']
            image = np.array(cjdata['image'], dtype=np.float32)
            mask = np.array(cjdata['tumorMask'], dtype=np.float32)
        
        # Resize to a smaller, consistent size
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        # Normalize image to 0-1 range
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Ensure mask is strictly binary (0 or 1) after resizing
        mask = (mask > 0.5).astype(np.float32)
        
        # Add channel dimension: (H, W) -> (1, H, W), PyTorch expects channels first
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)
        
        # Convert to PyTorch tensors
        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask)
        label = record["label"]  # 1=meningioma, 2=glioma, 3=pituitary
        
        return image_tensor, mask_tensor, label
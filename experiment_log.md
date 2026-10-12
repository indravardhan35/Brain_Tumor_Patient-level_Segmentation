# Experiment Log — Brain Tumour MRI Segmentation Project

## 2026-10-12 — Environment setup
- Confirmed laptop specs: NVIDIA GTX 1650, 16GB RAM, 167GB free storage
- Python 3.13.5 (VS Code kernel), with h5py, numpy, matplotlib, scikit-learn pre-installed
- Downloaded Figshare brain tumor dataset (Cheng et al., version 8)
- Verified dataset structure: 3064 .mat files across 4 folders, fields confirmed as cjdata.label, cjdata.PID, cjdata.image, cjdata.tumorMask
- Status: Complete

## 2026-10-12 — Single file read test
- Successfully read and displayed one sample (1.mat)
- Confirmed label, image shape, tumor mask shape, and decoded patient ID (e.g., "100360")
- Status: Complete

## 2026-10-12 — Dataset indexing
- Processed all 3064 .mat files, extracted filename, folder, label, and patient_id for each
- Saved as dataset_index.json
- Status: Complete, no errors

## 2026-10-12 — Patient-level split built and verified
- Grouped all 3064 images by patient ID: 233 unique patients confirmed
- Split patients (not images) 80/10/10: 186 train / 23 val / 24 test
- Verified zero overlap between train/val/test patient sets
- Resulting image counts: 2462 train / 280 val / 322 test
- Saved as patient_level_split.json
- Status: Complete, leak-free split confirmed
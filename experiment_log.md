# Experiment Log — Brain Tumour MRI Segmentation Project

## 2026-10-12 — Environment setup

* Confirmed laptop specs: NVIDIA GTX 1650, 16GB RAM, 167GB free storage
* Python 3.13.5 (VS Code kernel), with h5py, numpy, matplotlib, scikit-learn pre-installed
* Downloaded Figshare brain tumor dataset (Cheng et al., version 8)
* Verified dataset structure: 3064 .mat files across 4 folders, fields confirmed as cjdata.label, cjdata.PID, cjdata.image, cjdata.tumorMask
* Status: Complete

## 2026-10-12 — Single file read test

* Successfully read and displayed one sample (1.mat)
* Confirmed label, image shape, tumor mask shape, and decoded patient ID (e.g., "100360")
* Status: Complete

## 2026-10-12 — Dataset indexing

* Processed all 3064 .mat files, extracted filename, folder, label, and patient\_id for each
* Saved as dataset\_index.json
* Status: Complete, no errors

## 2026-10-12 — Patient-level split built and verified

* Grouped all 3064 images by patient ID: 233 unique patients confirmed
* Split patients (not images) 80/10/10: 186 train / 23 val / 24 test
* Verified zero overlap between train/val/test patient sets
* Resulting image counts: 2462 train / 280 val / 322 test
* Saved as patient\_level\_split.json
* Status: Complete, leak-free split confirmed



\## 2026-10-20 — PyTorch Dataset loader built and verified

\- Created dataset.py with BrainTumorDataset class (reads .mat files, resizes to 256x256, normalizes)

\- Verified dataset loads correctly: 2462 train images

\- Confirmed tensor shapes: image \[1,256,256], mask \[1,256,256]

\- Verified DataLoader batching works: batch shape \[4,1,256,256]

\- Status: Complete



\## 2026-10-20 — Model architecture built and verified

\- Created model.py with LightweightUNet: pretrained MobileNetV2 encoder + SE attention blocks in decoder

\- Modified first conv layer to accept 1-channel (grayscale) input instead of 3-channel RGB

\- Verified forward pass: input \[4,1,256,256] -> output \[4,1,256,256]

\- Total parameters: 5,565,569 (5.57M) - significantly lighter than Deep-MiSR's 13.79M

\- Status: Complete



\## 2026-10-22 — GPU setup fixed

\- Initial PyTorch install was CPU-only (cu121 tag unavailable for Python 3.13)

\- Reinstalled using cu124 CUDA build

\- Verified: CUDA available = True, Device = NVIDIA GeForce GTX 1650

\- Status: Complete



\## 2026-10-22 — First full training run (baseline, patient-level split)

\- Trained LightweightUNet (MobileNetV2 + SE attention) on full dataset: 2462 train / 280 val

\- Loss: CombinedLoss (0.5 Dice + 0.5 BCE), Adam optimizer, lr=1e-4, 20 epochs, batch size 8

\- Best Val Dice: 0.6922 (epoch 20)

\- Observed overfitting: train loss dropped to 0.039 while val Dice plateaued \~0.66-0.69 from epoch 10 onward

\- Comparison: Deep-MiSR reports 0.9281 Dice (random split) - gap expected at this stage, no augmentation/scheduling used yet

\- Status: Working baseline established, needs tuning (augmentation, LR scheduling) to close gap



\## 2026-10-22 — Training improvements: augmentation + LR scheduling + early stopping

\- Added random flip/rotation/brightness augmentation (training set only)

\- Added ReduceLROnPlateau scheduler (factor=0.5, patience=3)

\- Added early stopping (patience=7 epochs)

\- Retrained on full dataset: 2462 train / 280 val

\- Best Val Dice improved: 0.6922 -> 0.7320

\- Training stopped automatically at epoch 36 (early stopping triggered)

\- LR reduced 3 times during training (0.0001 -> 0.0000125)

\- Status: Improved baseline established



\## 2026-10-24 — Random split built (for leakage comparison)

\- Built random\_split.json: same 3064 images split randomly by image, not by patient

\- Measured actual leakage: 156/233 patients in both Train \& Val, 154/233 in both Train \& Test, 111/233 in both Val \& Test

\- This confirms, with real numbers, the data leakage problem described in Deep-MiSR's limitations

\- Status: Complete, strong evidence for patient-level split argument



\## 2026-10-24 — Random split training complete: leakage effect quantified

\- Trained identical model/hyperparameters on random\_split.json (same architecture, same code as patient-level run)

\- Best Val Dice (random split): 0.8236

\- Best Val Dice (patient-level split): 0.7320

\- Difference: +0.0916 (9.16 percentage points) attributable purely to data leakage

\- Confirms with direct, reproducible evidence that random splitting inflates reported performance

\- This is the project's core contribution #1, demonstrated empirically

\- Status: Complete - key result established


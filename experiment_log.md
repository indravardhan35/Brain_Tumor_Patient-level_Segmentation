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



\## 2026-10-24 — Per-class baseline established (patient-level model, test set)

\- Evaluated best\_model.pth on held-out test set (322 images, never used in training/val)

\- Per-class Dice: meningioma 0.9164 (n=103), glioma 0.6694 (n=117), pituitary 0.7567 (n=102)

\- Overall test Dice: 0.7761

\- Confirms glioma as weakest class, consistent with literature survey findings

\- This is the "before" baseline for the glioma-focused enhancement

\- Status: Complete



\## 2026-10-25 — Glioma-weighted loss attempt (class weight = 2.0 on BCE component)

\- Modified loss function: WeightedCombinedLoss applies 2x weight to glioma samples in the BCE term

\- Trained with identical architecture, augmentation, scheduler, early stopping as baseline

\- Overall Val Dice during training: 0.7423 (appeared better than baseline's 0.7320)

\- However, per-class TEST evaluation showed the opposite:

&#x20; - Meningioma: 0.9164 -> 0.9115 (flat)

&#x20; - Glioma: 0.6694 -> 0.6519 (WORSE, not better)

&#x20; - Pituitary: 0.7567 -> 0.7075 (worse)

&#x20; - Overall test Dice: 0.7761 -> 0.7526 (worse)

\- Conclusion: class-weighted BCE did not improve glioma segmentation; overall val Dice improvement was misleading (masked per-class degradation)

\- Root cause suspected: weighting whole-image loss is too blunt an instrument for glioma's actual problem (diffuse/ambiguous boundaries, not just underrepresentation)

\- Next step: try a boundary-aware loss (Tversky) instead of class-weighted BCE

\- Status: Negative result, informative, documented



\## 2026-10-25 — Tversky loss attempt (alpha=0.3, beta=0.7)

\- Combined Tversky (0.7 weight) + BCE (0.3 weight), beta>alpha to penalize false negatives more

\- Per-class test Dice: meningioma 0.9129, glioma 0.6578, pituitary 0.7850, overall 0.7797

\- Pituitary and overall improved vs baseline; glioma still did NOT improve (0.6578 vs 0.6694 baseline)

\- Two loss-function approaches now tried (weighted BCE, Tversky) - neither improved glioma specifically

\- Hypothesis: glioma's problem may be data-level (shape/size variability) rather than purely loss-weighting related

\- Next step: try oversampling glioma via WeightedRandomSampler, combined with Tversky loss

\- Status: Partial result, informative, glioma enhancement still unresolved



\## 2026-10-29 — Tversky + glioma oversampling (2.5x WeightedRandomSampler)

Config: LightweightUNet, patient-level split, CombinedTverskyLoss, augment=True

Result (test Dice): men 0.9036 | glioma 0.6478 | pit 0.7518 | overall 0.7626

vs baseline: glioma -0.0216, overall -0.0135

Status: NEGATIVE. Third failed glioma attempt; suggests the limit is boundary/representation, not class weighting.


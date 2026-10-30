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



\## 2026-10-29 — Tversky + glioma oversampling (2.5x WeightedRandomSampler)

Config: LightweightUNet, patient-level split, CombinedTverskyLoss, augment=True

Result (test Dice): men 0.9036 | glioma 0.6478 | pit 0.7518 | overall 0.7626

Comparison: glioma -0.0216 and overall -0.0135 vs baseline (0.6694 / 0.7761)

Status: NEGATIVE. Third loss/sampling attempt with no glioma gain.



\## 2026-10-29 — Glioma error analysis (baseline model, analyze\_glioma\_errors.py)

Config: best\_model.pth, patient-level test set, 117 glioma slices, 8 patients

Findings:

\- Per-patient glioma Dice ranges 0.54 to 0.84, so differences of about 0.02 between loss experiments are within noise.

\- Dice by tumour size: small 0.662, medium 0.628, large 0.718. Small tumours are over-segmented (recall 0.76 > precision 0.65); large tumours are under-segmented (precision 0.82 > recall 0.69).

\- 11 of 117 slices (9%) have Dice < 0.1, spread over 6 of 8 patients (MR033420 has 5). Without them, glioma Dice is about 0.74.

Comparison: n/a (diagnostic, no new model)

Status: DIAGNOSTIC. Failures are dominated by detection/localisation, not boundary fuzziness.



\## 2026-10-29 — Confidence check on misses (check\_confidence.py)

Config: same model; threshold sweep 0.5 / 0.3 / 0.2 on the 11 slices with Dice < 0.1

Findings:

\- 4 slices are blind (max\_prob 0.002 to 0.021, Dice 0 at every threshold).

\- 6 slices are confidently wrong (max\_prob about 1.0, Dice about 0): the model segments a structure other than the labelled tumour. 1 slice (max\_prob 0.746) is a partial wrong-place case.

\- Lowering the threshold recovers nothing, so under-confidence is ruled out and loss reweighting cannot fix these slices.

Status: DIAGNOSTIC. Hypothesis to test: mixed imaging planes / missing global context. Next: inspect the worst-case images, then choose the enhancement (ASPP multi-scale context vs boundary module), and consider patient-grouped k-fold CV.



\## 2026-10-29 — Visual inspection of the 11 glioma misses (Dice < 0.1)

Findings (manual review of miss\_montage.png):

\- Group A (idx 22, 29, 206, 236; 237 nearly empty): tumour not clearly visible; 3 of 4 blind slices are sagittal/coronal. Likely atypical/edge slices.

\- Group B (idx 13, 18, 19, 161, 260; 40 subtle): axial, tumour obvious, model confident but predicts adjacent or wrong region (localisation/context failure, not boundary fuzziness alone).

\- Only 4 of 11 failures are non-axial, so plane mixing alone does not explain them (plane distribution of the dataset still to be checked).

Status: DIAGNOSTIC. Next: ASPP multi-scale context module on bottleneck (Group B), with multi-seed or grouped-CV evaluation.



\## 2026-10-29 | Multi-seed baseline vs ASPP (patient-level split)

Config: train\_aspp.py, Adam 1e-3, bs16, ReduceLROnPlateau(0.5, p4), ES p10, seeds 0/1/2, --model {baseline,aspp}

Result (test Dice mean±std, men/gli/pit/overall):

&#x20; Baseline 0.9243±0.0053 / 0.6842±0.0231 / 0.7872±0.0125 / 0.7937±0.0129

&#x20; ASPP     0.9280±0.0021 / 0.6792±0.0075 / 0.7738±0.0092 / 0.7888±0.0039

Comparison: ASPP glioma -0.005 vs baseline, within seed std (0.023). Seed-0 gain (+0.0165) was noise. ASPP has 7.93M vs 5.57M params.

Status: NEGATIVE RESULT. ASPP not adopted as C2. Baseline seed-0 re-run differs from original train.py run (val 0.7448 vs 0.7320): script diff to be checked.



\## 2026-10-30 | 5-fold CV: patient-grouped vs random-slice (baseline, seed 0)

Config: train\_aspp.py recipe, 5 folds each, make\_cv\_folds.py (seed 42), bootstrap over 233 patients

Result: overall 0.7730 \[0.751, 0.794] (patient) vs 0.8087 \[0.790, 0.826] (random); gap +0.0357 \[0.027, 0.045]

&#x20; glioma 0.6757 vs 0.7278 (gap +0.0521 \[0.037, 0.068]); pituitary gap +0.0287; meningioma gap +0.0120

Comparison: earlier single-run gap (9.16 pts, val set, old recipe) was an overestimate; controlled gap is \~3.6 pts

Status: C1 confirmed with confidence intervals. Glioma per-patient Dice mean 0.646 (min 0.028, max 0.940).


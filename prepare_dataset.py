import h5py
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split

# ---- Step A: Collect info from every .mat file ----

data_folders = [
    "data/brainTumorDataPublic_1-766",
    "data/brainTumorDataPublic_767-1532",
    "data/brainTumorDataPublic_1533-2298",
    "data/brainTumorDataPublic_2299-3064",
]

records = []  # will hold one entry per image

for folder in data_folders:
    mat_files = [f for f in os.listdir(folder) if f.endswith(".mat")]
    print(f"Processing {folder}: {len(mat_files)} files")
    
    for fname in mat_files:
        file_path = os.path.join(folder, fname)
        try:
            with h5py.File(file_path, 'r') as f:
                cjdata = f['cjdata']
                label = int(np.array(cjdata['label']).item())
                pid_raw = np.array(cjdata['PID'])
                pid_string = ''.join(chr(int(c[0])) for c in pid_raw)
            
            records.append({
                "filename": fname,
                "folder": folder,
                "label": label,          # 1=meningioma, 2=glioma, 3=pituitary
                "patient_id": pid_string
            })
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

print(f"\nTotal records collected: {len(records)}")

# ---- Step B: Save this lookup table ----
with open("dataset_index.json", "w") as f:
    json.dump(records, f, indent=2)

print("Saved dataset_index.json")

# ---- Step C: Group by patient and split ----

from collections import defaultdict

# Group all image filenames by patient ID
patient_to_records = defaultdict(list)
for r in records:
    patient_to_records[r["patient_id"]].append(r)

unique_patients = list(patient_to_records.keys())
print(f"\nTotal unique patients: {len(unique_patients)}")

# Split PATIENTS (not images) into train/val/test: 80/10/10
train_patients, temp_patients = train_test_split(
    unique_patients, test_size=0.2, random_state=42
)
val_patients, test_patients = train_test_split(
    temp_patients, test_size=0.5, random_state=42
)

print(f"Train patients: {len(train_patients)}")
print(f"Val patients: {len(val_patients)}")
print(f"Test patients: {len(test_patients)}")

# ---- Step D: Verify no overlap between splits ----
train_set = set(train_patients)
val_set = set(val_patients)
test_set = set(test_patients)

overlap_train_val = train_set & val_set
overlap_train_test = train_set & test_set
overlap_val_test = val_set & test_set

print(f"\nOverlap check (should all be 0):")
print(f"Train-Val overlap: {len(overlap_train_val)}")
print(f"Train-Test overlap: {len(overlap_train_test)}")
print(f"Val-Test overlap: {len(overlap_val_test)}")

# ---- Step E: Build final image-level split lists ----
def get_records_for_patients(patient_list):
    result = []
    for p in patient_list:
        result.extend(patient_to_records[p])
    return result

train_records = get_records_for_patients(train_patients)
val_records = get_records_for_patients(val_patients)
test_records = get_records_for_patients(test_patients)

print(f"\nTrain images: {len(train_records)}")
print(f"Val images: {len(val_records)}")
print(f"Test images: {len(test_records)}")

# ---- Step F: Save the final split ----
split_data = {
    "train": train_records,
    "val": val_records,
    "test": test_records
}

with open("patient_level_split.json", "w") as f:
    json.dump(split_data, f, indent=2)

print("\nSaved patient_level_split.json")
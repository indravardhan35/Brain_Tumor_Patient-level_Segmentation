import json
from sklearn.model_selection import train_test_split

# Load the same dataset index we built earlier
with open("dataset_index.json", "r") as f:
    records = json.load(f)

print(f"Total records: {len(records)}")

# Split by IMAGE this time, not by patient, random slice-wise split
train_records, temp_records = train_test_split(records, test_size=0.2, random_state=42)
val_records, test_records = train_test_split(temp_records, test_size=0.5, random_state=42)

print(f"Train images: {len(train_records)}")
print(f"Val images: {len(val_records)}")
print(f"Test images: {len(test_records)}")

# Check how many patients leak across splits (for your report, this is the key evidence)
train_patients = set(r["patient_id"] for r in train_records)
val_patients = set(r["patient_id"] for r in val_records)
test_patients = set(r["patient_id"] for r in test_records)

leak_train_val = train_patients & val_patients
leak_train_test = train_patients & test_patients
leak_val_test = val_patients & test_patients

print(f"\nPatient leakage check:")
print(f"Patients in both Train & Val: {len(leak_train_val)}")
print(f"Patients in both Train & Test: {len(leak_train_test)}")
print(f"Patients in both Val & Test: {len(leak_val_test)}")

split_data = {
    "train": train_records,
    "val": val_records,
    "test": test_records
}

with open("random_split.json", "w") as f:
    json.dump(split_data, f, indent=2)

print("\nSaved random_split.json")
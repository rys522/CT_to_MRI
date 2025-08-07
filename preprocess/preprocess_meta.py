import nibabel as nib
import numpy as np
import h5py
import os
from pathlib import Path
import random
import pandas as pd

# --------- Root Paths ---------
input_root = Path("/fast_storage/intern1/Task1/brain")
output_root = Path("/fast_storage/intern1/CT_MRI_data/brain")
output_root.mkdir(parents=True, exist_ok=True)

# -------- Meta data ----------
excel_path = "/fast_storage/intern1/Task1/brain/overview/1_brain_train.xlsx"
mri_df = pd.read_excel(excel_path, sheet_name='MRI')


meta_dict = {}
for _, row in mri_df.iterrows():
    pid = row['ID']
    if pid not in meta_dict:
        meta_dict[pid] = {}
    meta_dict[pid]['MRI'] = row.to_dict()

# --------- Parameters ---------
train_ratio = 0.7
valid_ratio = 0.15
random.seed(42)

def normalize_ct(x):
    x = np.clip(x, -1000, 1000)
    return (x + 1000) / 2000

def normalize_mr(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-5)

def save_slices(slices, ct_path, mr_path, mask_path, seq_list, tesla_list):
    N, H, W = len(slices), slices[0][0].shape[0], slices[0][0].shape[1]

    with h5py.File(ct_path, "w") as f_ct, \
         h5py.File(mr_path, "w") as f_mr, \
         h5py.File(mask_path, "w") as f_mask:

        f_ct.create_dataset("data", (N, H, W), dtype=np.float32)
        f_mr.create_dataset("data", (N, H, W), dtype=np.float32)
        f_mask.create_dataset("data", (N, H, W), dtype=np.uint8)

        for i, (ct_s, mr_s, m_s) in enumerate(slices):
            f_ct["data"][i] = ct_s
            f_mr["data"][i] = mr_s
            f_mask["data"][i] = m_s

        f_ct.create_dataset("sequence", data=np.array(seq_list, dtype="S64"))
        f_ct.create_dataset("Tesla", data=np.array(tesla_list, dtype=np.float32))


def center_crop_or_pad(x, target_shape=(512, 512)):
    H, W = x.shape
    target_H, target_W = target_shape
    pad_h = max(0, target_H - H)
    pad_w = max(0, target_W - W)
    pad_before_h = pad_h // 2
    pad_after_h = pad_h - pad_before_h
    pad_before_w = pad_w // 2
    pad_after_w = pad_w - pad_before_w

    x_padded = np.pad(x, ((pad_before_h, pad_after_h), (pad_before_w, pad_after_w)), mode='constant')

    # crop if too big
    H_p, W_p = x_padded.shape
    start_h = (H_p - target_H) // 2
    start_w = (W_p - target_W) // 2
    return x_padded[start_h:start_h+target_H, start_w:start_w+target_W]

# --------- Step 1: Collect all subjects ---------
all_subjects = sorted([subdir for subdir in input_root.iterdir() if subdir.is_dir()])
random.shuffle(all_subjects)

# --------- Step 2: Split subjects ---------
n_total = len(all_subjects)
n_train = int(n_total * train_ratio)
n_valid = int(n_total * valid_ratio)
n_test = n_total - n_train - n_valid

train_subjects = all_subjects[:n_train]
valid_subjects = all_subjects[n_train:n_train+n_valid]
test_subjects  = all_subjects[n_train+n_valid:]

print(f"Subjects: {n_total} total → {n_train} train / {n_valid} valid / {n_test} test")

# --------- Step 3: Helper to collect slices ---------
def load_subject_slices(subdir):
    ct_path = subdir / "ct.nii.gz"
    mr_path = subdir / "mr.nii.gz"
    mask_path = subdir / "mask.nii.gz"

    if not (ct_path.exists() and mr_path.exists() and mask_path.exists()):
        print(f"[{subdir.name}] Skipped (missing files)")
        return []

    ct = nib.load(str(ct_path)).get_fdata()
    mr = nib.load(str(mr_path)).get_fdata()
    mask = nib.load(str(mask_path)).get_fdata()

    ct *= mask
    mr *= mask

    ct = normalize_ct(ct)
    mr = normalize_mr(mr)

    slices = []
    for i in range(ct.shape[2]):
        if np.max(ct[:, :, i]) == 0 or np.max(mr[:, :, i]) == 0:
            continue
        ct_slice = center_crop_or_pad(ct[:, :, i])
        mr_slice = center_crop_or_pad(mr[:, :, i])
        mask_slice = center_crop_or_pad(mask[:, :, i])
        slices.append((ct_slice, mr_slice, mask_slice.astype(np.uint8)))

    print(f"[{subdir.name}] {len(slices)} valid slices")
    return slices

# --------- Step 4: Aggregate & Save per split ---------
for name, subject_list in zip(["train", "valid", "test"], [train_subjects, valid_subjects, test_subjects]):
    all_slices = []
    all_sequence = []
    all_tesla = []

    for subdir in subject_list:
        pid = subdir.name
        slices = load_subject_slices(subdir)
        if not slices:
            continue

        meta = meta_dict.get(pid, {})
        def safe_float(x): 
            try: return float(x)
            except: return 0.0

        seq = meta.get("SeriesDescription", "")
        tesla = meta.get("MagneticFieldStrength", "")

        all_slices.extend(slices)
        all_sequence.extend([seq] * len(slices))
        all_tesla.extend([tesla] * len(slices))

    print(f"[{name.upper()}] Total slices: {len(all_slices)}")
    save_slices(
        all_slices,
        output_root / f"{name}_ct.h5",
        output_root / f"{name}_mri.h5",
        output_root / f"{name}_mask.h5",
        seq_list=all_sequence,
        tesla_list=all_tesla,
    )
    print(f"[{name.upper()}] Saved to {output_root}/{name}_ct.h5, {name}_mri.h5 and {name}_mask.h5\n")
import nibabel as nib
import numpy as np
import h5py
import os
from pathlib import Path
import random
from nipype.interfaces.fsl import FAST
import tempfile

from concurrent.futures import ProcessPoolExecutor, as_completed

import logging

# --------- Logger ---------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --------- Root Paths ---------
input_root = Path("/fast_storage/intern1/Task1/brain")
output_root = Path("/fast_storage/intern1/CT_MRI_data_256_256/brain")
output_root.mkdir(parents=True, exist_ok=True)
preview_root = Path(__file__).resolve().parent / "preview"

# --------- Parameters ---------
AXES = [2]  # sagittal, coronal, axial
IMG_CROP = (256, 256)
SLICES = 200 # num of middle slices
train_ratio = 0.7
valid_ratio = 0.15
random.seed(42)
num_workers= 6

def run_fast_worker(mr_slice):
    try:
        tissue = run_fast_on_slice(mr_slice)
        return tissue, None
    except Exception as e:
        return np.zeros((3, mr_slice.shape[0], mr_slice.shape[1]), dtype=np.float32), str(e)

def run_fast_on_slice(slice_np: np.ndarray) -> np.ndarray:
    """Run FSL FAST on a 2D slice and return (3, H, W) tissue probability map."""
    H, W = slice_np.shape
    with tempfile.TemporaryDirectory() as tmpdir:
        input_nii = os.path.join(tmpdir, "input.nii.gz")
        nib.save(nib.Nifti1Image(slice_np[np.newaxis, :, :], affine=np.eye(4)), input_nii)

        fast = FAST(in_files=input_nii, no_bias=True, segments=True, probability_maps=True)
        fast.base_output_dir = tmpdir
        result = fast.run()

        # Read pveseg0, pveseg1, pveseg2 (CSF, GM, WM)
        pve_maps = []
        for i in range(3):
            pve_path = os.path.join(tmpdir, f"input_pve_{i}.nii.gz")
            pve_data = nib.load(pve_path).get_fdata()
            pve_maps.append(pve_data[0])  # drop first dim

        return np.stack(pve_maps, axis=0)  # shape: (3, H, W)
    
def normalize_ct(x):
    x = np.clip(x, -1000, 1000)
    return (x + 1000) / 2000

def normalize_mr(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-5)

def save_slices(slices, ct_path, mr_path, mask_path, preview_dir=None, num_workers=num_workers):
    N, H, W = len(slices), slices[0][0].shape[0], slices[0][0].shape[1]
    logger.info(f"Start saving {N} slices to:")
    logger.info(f"       - CT: {ct_path}")
    logger.info(f"       - MR: {mr_path}")
    logger.info(f"       - MASK: {mask_path}")

    with h5py.File(ct_path, "w") as f_ct, \
         h5py.File(mr_path, "w") as f_mr, \
         h5py.File(mask_path, "w") as f_mask:

        f_ct.create_dataset("data", (N, H, W), dtype=np.float32)
        f_mr.create_dataset("data", (N, H, W), dtype=np.float32)
        f_mask.create_dataset("data", (N, H, W), dtype=np.uint8)
        f_mr.create_dataset("tissue_map", (N, 3, H, W), dtype=np.float32)

        rand_idx = random.randint(0, N - 1)
        mr_slices = [mr_s for _, mr_s, _ in slices]

        logger.info(f"Running FAST with {num_workers} processes ...")
        tissues = []
        errors = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {executor.submit(run_fast_worker, mr_s): i for i, mr_s in enumerate(mr_slices)}
            done = 0
            total = len(future_to_idx)
            tissues = [None] * total
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                tissue, err = future.result()
                done += 1
                logger.info(f"FAST {done}/{total} slices complete")
                tissues[idx] = (idx, tissue, err)
        for i, (ct_s, mr_s, m_s) in enumerate(slices):
            logger.info(f"[{i+1:>4}/{N}] Processing slice...")

            f_ct["data"][i] = ct_s
            f_mr["data"][i] = mr_s
            f_mask["data"][i] = m_s

            _, tissue, err = tissues[i]
            f_mr["tissue_map"][i] = tissue

            if i == rand_idx and preview_dir is not None:
                preview_dir.mkdir(parents=True, exist_ok=True)
                affine = np.eye(4)
                nib.save(nib.Nifti1Image(ct_s, affine), preview_dir / "ct.nii.gz")
                nib.save(nib.Nifti1Image(mr_s, affine), preview_dir / "mri.nii.gz")
                for j, label in enumerate(["CSF", "GM", "WM"]):
                    nib.save(nib.Nifti1Image(tissue[j], affine), preview_dir / f"tissue_{label}.nii.gz")
                logger.info(f"       [✓] Preview slice saved at: {preview_dir}")

    logger.info(f"All {N} slices saved successfully.\n")

def center_crop_or_pad(x, target_shape=IMG_CROP):
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

logger.info(f"Subjects: {n_total} total → {n_train} train / {n_valid} valid / {n_test} test")

# --------- Step 3: Helper to collect slices ---------
def load_subject_slices(subdir):
    ct_path = subdir / "ct.nii.gz"
    mr_path = subdir / "mr.nii.gz"
    mask_path = subdir / "mask.nii.gz"

    if not (ct_path.exists() and mr_path.exists() and mask_path.exists()):
        logger.warning(f"[{subdir.name}] Skipped (missing files)")
        return []

    ct = nib.load(str(ct_path)).get_fdata()
    mr = nib.load(str(mr_path)).get_fdata()
    mask = nib.load(str(mask_path)).get_fdata()

    ct *= mask
    mr *= mask

    ct = normalize_ct(ct)
    mr = normalize_mr(mr)

    slices = []
    for axis in AXES:
        num_slices = ct.shape[axis]
        if num_slices < SLICES:
            start = 0
            end = num_slices
        else:
            start = (num_slices - SLICES) // 2
            end = start + SLICES
        for i in range(start, end):
            
            if axis == 2:
                ct_slice = center_crop_or_pad(ct[:, :, i])
                mr_slice = center_crop_or_pad(mr[:, :, i])
                mask_slice = center_crop_or_pad(mask[:, :, i])
            elif axis == 1:
                ct_slice = center_crop_or_pad(ct[:, i, :])
                mr_slice = center_crop_or_pad(mr[:, i, :])
                mask_slice = center_crop_or_pad(mask[:, i, :])
            elif axis == 0:
                ct_slice = center_crop_or_pad(ct[i, :, :])
                mr_slice = center_crop_or_pad(mr[i, :, :])
                mask_slice = center_crop_or_pad(mask[i, :, :])
            else:
                raise ValueError("axis must be 0, 1, or 2")

            if np.max(ct_slice) == 0 or np.max(mr_slice) == 0:
                continue

            slices.append((ct_slice, mr_slice, mask_slice.astype(np.uint8)))

    logger.info(f"[{subdir.name}] {len(slices)} valid slices")
    return slices

# --------- Step 4: Aggregate & Save per split ---------
for name, subject_list in zip(["train", "valid", "test"], [train_subjects, valid_subjects, test_subjects]):
    all_slices = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(load_subject_slices, subdir): subdir for subdir in subject_list}
        for future in as_completed(futures):
            subdir = futures[future]
            try:
                slices = future.result()
                all_slices.extend(slices)
                logger.info(f"[{subdir.name}] Slices loaded and added.")
            except Exception as e:
                logger.error(f"[{subdir.name}] ERROR: {e}")

    logger.info(f"[{name.upper()}] Total slices: {len(all_slices)}")
    save_slices(
        all_slices,
        output_root / f"{name}_ct.h5",
        output_root / f"{name}_mri.h5",
        output_root / f"{name}_mask.h5"
    )
    logger.info(f"[{name.upper()}] Saved to {output_root}/{name}_ct.h5, {name}_mri.h5 and {name}_mask.h5\n")
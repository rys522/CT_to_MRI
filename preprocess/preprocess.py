import nibabel as nib
import numpy as np
import h5py
import os
from pathlib import Path
import random
from nipype.interfaces import fsl
import tempfile

from concurrent.futures import ProcessPoolExecutor

import logging

# --------- Logger ---------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# --------- Root Paths ---------
INPUT_ROOT = Path("/fast_storage/intern1/Task1/brain")
OUTPUT_ROOT = Path("/fast_storage/intern1/CT_MRI_data_3d/brain")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
NUM_WORKERS = 5
random.seed(42)


def process_subject_volume(subdir: Path):
    """
    Load and preprocess a subject's 3D CT/MRI volumes and mask.
    Returns (ct_vol, mri_reg_vol, mask_vol, subject_name) or None on failure.
    """
    ct_file = subdir / "ct.nii.gz"
    mr_file = subdir / "mr.nii.gz"
    if not (ct_file.exists() and mr_file.exists()):
        logger.warning(f"[{subdir.name}] Missing CT/MR → skip")
        return None

    # load volumes
    ct_vol = nib.load(str(ct_file)).get_fdata().astype(np.float32)
    mr_vol = nib.load(str(mr_file)).get_fdata().astype(np.float32)

    # affine registration: MR → CT
    tmp_reg = tempfile.mkdtemp(prefix=subdir.name + "_")
    mr_reg_file = Path(tmp_reg) / "mr_reg.nii.gz"
    mat_file = Path(tmp_reg) / "mr2ct.mat"

    flirt = fsl.FLIRT(
        in_file=str(mr_file),
        reference=str(ct_file),
        out_file=str(mr_reg_file),
        out_matrix_file=str(mat_file),
    )
    flirt.run()
    mr_reg_vol = nib.load(str(mr_reg_file)).get_fdata().astype(np.float32)

    # skull-strip (BET)
    bet = fsl.BET(
        in_file=str(mr_reg_file),
        out_file=str(Path(tmp_reg) / "mr_reg_brain.nii.gz"),
        mask=True,
        frac=0.5,
    )
    res_bet = bet.run()
    mask_file = Path(res_bet.outputs.mask_file)

    if mask_file.stat().st_size == 0:
        logger.error(f"[{subdir.name}] Empty BET mask → skip")
        return None

    mask_vol = nib.load(str(mask_file)).get_fdata().astype(np.uint8)

    # cleanup temporary files
    for p in [mr_reg_file, mat_file, mask_file]:
        try:
            os.remove(str(p))
        except:
            pass
    try:
        os.rmdir(tmp_reg)
    except:
        pass

    return ct_vol, mr_reg_vol, mask_vol, subdir.name


def save_volumes(volumes, out_dir: Path, split_name: str):
    """
    Save lists of (ct_vol, mri_vol, mask_vol, subj) as HDF5 files per split.
    """
    if not volumes:
        logger.warning(f"No volumes to save for {split_name}")
        return

    # stack arrays: shape (N, D, H, W)
    cts, mris, masks, subs = zip(*volumes)
    ct_arr = np.stack(cts, axis=0)
    mri_arr = np.stack(mris, axis=0)
    mask_arr = np.stack(masks, axis=0)

    out_path = out_dir / f"{split_name}_volumes.h5"
    with h5py.File(out_path, "w") as f:
        f.create_dataset("ct", data=ct_arr, compression="gzip")
        f.create_dataset("mri", data=mri_arr, compression="gzip")
        f.create_dataset("mask", data=mask_arr, compression="gzip")
        # subject names
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("subject", data=np.array(subs, dtype=dt))

    logger.info(f"Saved {split_name} volumes to {out_path}")


def main():
    all_subj = sorted([d for d in INPUT_ROOT.iterdir() if d.is_dir()])
    random.shuffle(all_subj)
    n = len(all_subj)
    n_tr = int(n * TRAIN_RATIO)
    n_val = int(n * VALID_RATIO)

    splits = {
        "train": all_subj[:n_tr],
        "valid": all_subj[n_tr : n_tr + n_val],
        "test": all_subj[n_tr + n_val :],
    }

    for split_name, subj_list in splits.items():
        logger.info(f"Processing {split_name.upper()} split: {len(subj_list)} subjects")
        volumes = []
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as exe:
            for res in exe.map(process_subject_volume, subj_list):
                if res is not None:
                    volumes.append(res)
        save_volumes(volumes, OUTPUT_ROOT, split_name)


if __name__ == "__main__":
    main()

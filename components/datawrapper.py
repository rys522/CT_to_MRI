import random
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Optional

from params import GeneralConfig
from common.logger import logger


def mask_soft_tissue_only(
    ct: np.ndarray, mask: np.ndarray, min_hu: float = -50, max_hu: float = 50
) -> np.ndarray:

    mask_hu = (ct >= min_hu) & (ct <= max_hu)
    ct_masked_hu = np.where(mask_hu, ct, min_hu)
    ct_masked = np.where(mask, ct_masked_hu, min_hu)

    return ct_masked


def sigmoid_soft_clip(ct_norm: np.ndarray, scale: float = 0.1) -> np.ndarray:

    centered = (ct_norm - 0.5) / scale
    ct_sigmoid = 1 / (1 + np.exp(-centered))
    return ct_sigmoid


def norm(x, mask):
    mask_region = x[mask > 0]
    if mask_region.size == 0:
        return x
    max_val = np.max(mask_region)
    min_val = np.min(mask_region)
    x_norm = np.zeros_like(x, dtype=np.float32)
    x_norm[mask > 0] = (mask_region - min_val) / (max_val - min_val + 1e-8)
    return x_norm


def norm_mri(mri: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask is not None:
        region = mri[mask > 0]
        if region.size == 0:
            return mri
        std = np.std(region)
        if std > 0:
            mri_norm = mri.copy()
            mri_norm[mask > 0] = region / (std + 1e-8)
            mri_norm[mask == 0] = 0
            return mri_norm
        else:
            mri_out = mri.copy()
            mri_out[mask == 0] = 0
            return mri_out
    else:
        std = np.std(mri)
        if std > 0:
            return mri / (std)
        return mri


def check_nan(arr, name=""):
    if np.isnan(arr).any():
        logger.debug(f"⚠️ NaN detected in {name} (shape={arr.shape})")
        return -1
    else:
        return 0


def is_all_zero(x: np.ndarray) -> bool:
    return np.count_nonzero(x) == 0


def collate_remove_none(batch):
    batch = [b for b in batch if b is not None and not any(x is None for x in b)]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


class DataWrapper(Dataset):
    IMG = 0
    COND = 1
    MASK = 2

    def __init__(
        self,
        file_path: str,
        config: GeneralConfig,
        training_mode: bool = True,
        debug_mode: bool = False,
    ):
        super().__init__()

        self.config = config

        with h5py.File(file_path, "r") as f:
            self.mri = f["mri"][:]
            self.ct = f["ct"][:]
            self.mask = f["mask"][:] if "mask" in f else None

        assert self.mri.shape == self.ct.shape, "MRI and CT shapes must match"
        if self.mask is not None:
            assert self.mask.shape == self.mri.shape, "Mask shape must match MRI/CT"

        n, x_dim, y_dim, z_dim = self.ct.shape

        ct_clip = self.config.ct_clip
        scale = self.config.scale

        self.num_subj = n

        if self.config.axes == "z":
            self.num_slices = z_dim
        elif self.config.axes == "y":
            self.num_slices = y_dim
        elif self.config.axes == "x":
            self.num_slices = x_dim
        else:
            raise ValueError(f"Unknown axis {self.config.axes!r}")

        self.valid_slices = []

        for subj_idx in range(n):
            mri = self.mri[subj_idx, :, :, :]
            ct = self.ct[subj_idx, :, :, :]
            mask = self.mask[subj_idx, :, :, :] if self.mask is not None else None

            if mask is not None:
                mask = (mask > 0).astype(np.float32)
                mri = mri * mask

            ct_hard = mask_soft_tissue_only(
                ct, mask, min_hu=ct_clip[0], max_hu=ct_clip[1]
            )
            ct_hard = norm(ct_hard, mask)
            ct_soft = sigmoid_soft_clip(ct_hard, scale=scale)

            mri = norm_mri(mri, mask)
            ct = norm(ct_soft, mask)

            for slice_idx in range(self.num_slices):
                mri_slice, ct_slice, mask_slice = self._get_slice_arrays(
                    slice_idx, mri, ct, mask
                )
                if (
                    check_nan(mri_slice, "MRI slice") < 0
                    or check_nan(ct_slice, "CT slice") < 0
                    or (
                        mask_slice is not None
                        and check_nan(mask_slice, "Mask slice") < 0
                    )
                ):
                    continue
                if (
                    is_all_zero(mri_slice)
                    or is_all_zero(ct_slice)
                    or (mask_slice is not None and is_all_zero(mask_slice))
                ):
                    continue
                self.valid_slices.append((mri_slice, ct_slice, mask_slice))

        self.training_mode = training_mode
        if debug_mode:
            self.valid_slices = self.valid_slices[: (1000 if training_mode else 100)]

        self.data_len = len(self.valid_slices)

    def _get_slice_arrays(self, slice_idx, mri, ct, mask):
        if self.config.axes == "z":
            return (
                np.rot90(mri[:, :, slice_idx], k=3).copy(),
                np.rot90(ct[:, :, slice_idx], k=3).copy(),
                (
                    np.rot90(mask[:, :, slice_idx], k=3).copy()
                    if mask is not None
                    else None
                ),
            )
        elif self.config.axes == "y":
            return (
                np.rot90(mri[:, slice_idx, :], k=1).copy(),
                np.rot90(ct[:, slice_idx, :], k=1).copy(),
                (
                    np.rot90(mask[:, slice_idx, :], k=1).copy()
                    if mask is not None
                    else None
                ),
            )
        elif self.config.axes == "x":
            return (
                np.rot90(mri[slice_idx, :, :], k=1).copy(),
                np.rot90(ct[slice_idx, :, :], k=1).copy(),
                (
                    np.rot90(mask[slice_idx, :, :], k=1).copy()
                    if mask is not None
                    else None
                ),
            )

    def __getitem__(self, idx):

        mri_slice, ct_slice, mask_slice = self.valid_slices[idx]

        mri = torch.from_numpy(mri_slice).float().unsqueeze(0)
        ct = torch.from_numpy(ct_slice).float().unsqueeze(0)

        if mask_slice is not None:
            mask = torch.from_numpy(mask_slice).float().unsqueeze(0)
        else:
            mask = None

        # Augmentation
        if self.training_mode:
            if random.random() > 0.5:
                mri = torch.flip(mri, dims=[1])
                ct = torch.flip(ct, dims=[1])
                if mask is not None:
                    mask = torch.flip(mask, dims=[1])
            if random.random() > 0.5:
                mri = torch.flip(mri, dims=[2])
                ct = torch.flip(ct, dims=[2])
                if mask is not None:
                    mask = torch.flip(mask, dims=[2])

        if mask is not None:
            return mri, ct, mask
        else:
            return mri, ct

    def __len__(self):
        return self.data_len


def get_data_wrapper_loader(
    file_path: str,
    config: GeneralConfig,
    training_mode: bool = True,
    batch: int = 1,
    num_workers: int = 4,
    shuffle: bool = True,
    debug_mode: bool = False,
):

    dataset = DataWrapper(
        file_path=file_path,
        config=config,
        training_mode=training_mode,
        debug_mode=debug_mode,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=shuffle,
        collate_fn=collate_remove_none,
    )

    return dataloader, dataset, len(dataset)

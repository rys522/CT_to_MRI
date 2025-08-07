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


def norm(x):
    if np.max(x) > 0:
        return (x - np.min(x)) / (np.max(x) - np.min(x))
    return x


def norm_mri(mri: np.ndarray) -> np.ndarray:
    std = np.std(mri)
    if std > 0:
        return mri / std
    return mri


def check_nan(arr, name=""):
    if np.isnan(arr).any():
        logger.debug(f"⚠️ NaN detected in {name} (shape={arr.shape})")
        return -1


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
        file_path: tuple[str, str],
        config: GeneralConfig,
        mask_path: Optional[str] = None,
        training_mode: bool = True,
        debug_mode: bool = False,
    ):
        super().__init__()
        mri_path, ct_path = file_path

        self.config = config

        with h5py.File(mri_path, "r") as f:
            self.mri = f["data"][:]
        with h5py.File(ct_path, "r") as f:
            self.ct = f["data"][:]

        if mask_path:
            with h5py.File(mask_path, "r") as f:
                self.mask = f["data"][:]
        else:
            self.mask = None

        assert self.mri.shape == self.ct.shape, "MRI and CT shapes must match"
        if self.mask is not None:
            assert self.mask.shape == self.mri.shape, "Mask shape must match MRI/CT"

        n, x_dim, y_dim, z_dim = self.ct.shape

        ct_clip = self.config.ct_clip
        scale = self.config.scale

        for subj_idx in range(n):
            mri = np.rot90(self.mri[subj_idx, :, :, :], k=3, axes=(0, 1))
            ct = np.rot90(self.ct[subj_idx, :, :, :], k=3, axes=(0, 1))
            mask = (
                np.rot90(self.mask[subj_idx, :, :, :], k=3, axes=(0, 1))
                if self.mask is not None
                else None
            )
            if mask is not None:
                mask = (mask > 0).astype(np.float32)
                mri = mri * mask

            ct_slice_hard = mask_soft_tissue_only(
                ct, mask, min_hu=ct_clip[0], max_hu=ct_clip[1]
            )

            ct_slice_hard = norm(ct_slice_hard)

            ct_slice_soft = sigmoid_soft_clip(ct_slice_hard, scale=scale)

            mri = norm_mri(mri)
            ct = norm(ct_slice_soft)

            self.ct[subj_idx, :, :, :] = np.nan_to_num(ct, nan=0.0)
            self.mri[subj_idx, :, :, :] = np.nan_to_num(mri, nan=0.0)

        self.training_mode = training_mode

        n, x_dim, y_dim, z_dim = self.ct.shape
        self.num_subj = n

        if self.config.axes == "z":
            self.num_slices = z_dim
        elif self.config.axes == "y":
            self.num_slices = y_dim
        elif self.config.axes == "x":
            self.num_slices = x_dim
        else:
            raise ValueError(f"Unknown axis {self.config.axes!r}")

        total = self.num_subj * self.num_slices
        if debug_mode:
            total = min(total, 1000 if training_mode else 100)

        self.data_len = total

        self.blur = transforms.GaussianBlur(kernel_size=7)

    def __getitem__(self, idx):

        subj_idx = idx // self.num_slices
        slice_idx = idx % self.num_slices

        mri = self.mri[subj_idx, :, :, :]
        ct = self.ct[subj_idx, :, :, :]
        mask = self.mask[subj_idx, :, :, :] if self.mask is not None else None

        if self.config.axes == "z":
            mri_slice = mri[:, :, slice_idx]
            ct_slice = ct[:, :, slice_idx]
            mask_slice = mask[:, :, slice_idx] if self.mask is not None else None

        elif self.config.axes == "y":
            # shape (X, Z)
            mri_slice = mri[:, slice_idx, :]
            ct_slice = ct[:, slice_idx, :]
            mask_slice = mask[:, slice_idx, :] if self.mask is not None else None

        elif self.config.axes == "x":
            # shape (Y, Z)
            mri_slice = mri[slice_idx, :, :]
            ct_slice = ct[slice_idx, :, :]
            mask_slice = mask[slice_idx, :, :] if self.mask is not None else None

        mri = torch.from_numpy(mri_slice).float().unsqueeze(0)
        ct = torch.from_numpy(ct_slice).float().unsqueeze(0)

        if mask is not None:
            mask = torch.from_numpy(mask_slice).float().unsqueeze(0)

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
    file_path: tuple[str, str] | tuple[str, str, str],
    config: GeneralConfig,
    training_mode: bool = True,
    batch: int = 1,
    num_workers: int = 4,
    shuffle: bool = True,
    debug_mode: bool = False,
):

    mask_path = file_path[2] if len(file_path) == 3 else None

    dataset = DataWrapper(
        file_path=file_path[:2],
        config=config,
        mask_path=mask_path,
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

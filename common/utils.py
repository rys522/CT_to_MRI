import os
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.nn import functional


def timestamp() -> str:
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")


def separator(cols: int = 100) -> str:
    return "#" * cols


def seconds_to_dhms(seconds: float) -> str:
    s = seconds % 60
    m = (seconds // 60) % 60
    h = seconds // (60 * 60) % 1000
    return f"{int(h):02}h {int(m):02}m {int(s):02}s"


def call_next_id(run_dir: Path) -> int:
    run_ids = []
    os.makedirs(run_dir, exist_ok=True)
    for entry in os.listdir(run_dir):
        if (run_dir / entry).is_dir():
            try:
                run_ids.append(int(entry.split("_")[0]))
            except ValueError:
                continue
    return max(run_ids, default=-1) + 1


class SSIMcal(torch.nn.Module):
    def __init__(
        self,
        win_size: int = 7,
        k1: float = 0.01,
        k2: float = 0.03,
    ):
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size**2)
        np = win_size**2
        self.cov_norm = np / (np - 1)

    def forward(
        self,
        img: torch.Tensor,
        ref: torch.Tensor,
        data_range: torch.Tensor,
        return_map: bool = False, 

    ) -> torch.Tensor:
        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2

        padding = self.win_size // 2

        ux = functional.conv2d(img, self.w.to(img.device), padding=padding)
        uy = functional.conv2d(ref, self.w.to(img.device), padding=padding)
        uxx = functional.conv2d(img * img, self.w.to(img.device), padding=padding)
        uyy = functional.conv2d(ref * ref, self.w.to(img.device), padding=padding)
        uxy = functional.conv2d(img * ref, self.w.to(img.device), padding=padding)

        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)

        A1 = 2 * ux * uy + C1
        A2 = 2 * vxy + C2
        B1 = ux**2 + uy**2 + C1
        B2 = vx + vy + C2

        S = (A1 * A2) / (B1 * B2)
        if return_map:
            return S  # [B, 1, H, W]
        else:
            return torch.mean(S, dim=[2, 3], keepdim=True)


ssim_cal = SSIMcal()

IMG_DIM: int = 4


def calculate_ssim(
    img: torch.Tensor,
    ref: torch.Tensor,
    mask: torch.Tensor | None = None,
    
) -> torch.Tensor:
    if not (img.dim() == IMG_DIM and ref.dim() == IMG_DIM):
        raise ValueError("All tensors must be 4D.")

    if mask is not None and (mask.dim() != IMG_DIM):
        raise ValueError("Mask must be 4D.")

    if img.shape[1] == 2:
        img = torch.sqrt(img[:, :1, ...] ** 2 + img[:, 1:, ...] ** 2)
        ref = torch.sqrt(ref[:, :1, ...] ** 2 + ref[:, 1:, ...] ** 2)

    ones = torch.ones(ref.shape[0], device=ref.device)

    if mask is None:
        img_mask = img
        ref_mask = ref
        ssim = ssim_cal(img_mask, ref_mask, ones)
    else:
        if mask.shape[1] == 2:
            mask = torch.sqrt(mask[:, :1, ...] ** 2 + mask[:, 1:, ...] ** 2)
        ssim_map = ssim_cal(img, ref, ones, return_map=True)
        ssim = (ssim_map * mask).sum(dim=(1, 2, 3), keepdim=True) / mask.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)

    return ssim


def calculate_psnr(
    img: torch.Tensor,
    ref: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if not (img.dim() == IMG_DIM and ref.dim() == IMG_DIM):
        raise ValueError("All tensors must be 4D.")

    if mask is not None and (mask.dim() != IMG_DIM):
        raise ValueError("Mask must be 4D.")

    if img.shape[1] == 2:
        img = torch.sqrt(img[:, :1, ...] ** 2 + img[:, 1:, ...] ** 2)
        ref = torch.sqrt(ref[:, :1, ...] ** 2 + ref[:, 1:, ...] ** 2)

    diff_sq = (img - ref) ** 2

    if mask is not None:
        if mask.shape[1] == 2:
            mask = torch.sqrt(mask[:, :1, ...] ** 2 + mask[:, 1:, ...] ** 2)
        masked_diff = diff_sq * mask
        mask_sum = mask.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
        mse = masked_diff.sum(dim=(1, 2, 3), keepdim=True) / mask_sum
    else:
        mse = diff_sq.mean(dim=(1, 2, 3), keepdim=True)

    img_max = torch.amax(ref, dim=(1, 2, 3), keepdim=True)

    psnr = 10 * torch.log10((img_max**2 + 1e-12) / (mse + 1e-12))
    return psnr

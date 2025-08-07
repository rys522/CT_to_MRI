import torch
import torch.nn as nn
import torch.nn.functional as F

from params import GMM_ModelConfig, config


def norm_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Min-max normalize tensor to [0, 1]
    """
    min_val = x.amin(dim=(-2, -1), keepdim=True)
    max_val = x.amax(dim=(-2, -1), keepdim=True)
    return (x - min_val) / (max_val - min_val + 1e-5)

def mask_soft_tissue_only_tensor(ct: torch.Tensor) -> torch.Tensor:
    """
    Clip soft tissue range in normalized HU (0~1).
    Replace values outside the range with norm_min.
    """
    min_hu = config.hu_min
    max_hu = config.hu_max

    norm_min = (min_hu + 1000) / 2000
    norm_max = (max_hu + 1000) / 2000

    mask = (ct >= norm_min) & (ct <= norm_max)
    ct_masked = torch.where(mask, ct, torch.tensor(norm_min, dtype=ct.dtype, device=ct.device))

    ct_out = norm_tensor(ct_masked)

    return ct_out


class LearnableGMMDecomposer(nn.Module):
    def __init__(self, device: torch.device, modelconfig: GMM_ModelConfig):
        super().__init__()
        self.device = device
        self.num_tissues = modelconfig.num_tissues
        self.tissue_names = list(modelconfig.tissue_dict.keys())
        known_num = len(self.tissue_names)
        self.hu_max = config.ct_clip[1]
        self.hu_min = config.ct_clip[0]
        # Normalize HU mean/std to 0~1
        def hu2norm(x):
            return (x - self.hu_min) / (self.hu_max - self.hu_min)

        def std_hu2norm(std):
            return std / (self.hu_max - self.hu_min)
        
        # Initialize means
        known_means = [hu2norm(modelconfig.tissue_dict[name]) for name in self.tissue_names]
        if self.num_tissues > known_num:
            extra_count = self.num_tissues - known_num
            # extra means in 0~1
            extra_means = torch.linspace(0, 0, steps=extra_count).tolist()
            all_means = known_means + extra_means
        else:
            all_means = known_means[:self.num_tissues]

        init_means = torch.tensor(all_means, dtype=torch.float32).view(1, self.num_tissues, 1, 1).to(device)
        self.mu = nn.Parameter(init_means)

        # Initialize stds (scale to 0~1)
        known_stds = [std_hu2norm(modelconfig.std_dict[name]) for name in self.tissue_names]
        if self.num_tissues > known_num:
            extra_stds = [std_hu2norm(40.0)] * (self.num_tissues - known_num)
            all_stds = known_stds + extra_stds
        else:
            all_stds = known_stds[:self.num_tissues]

        init_stds = torch.tensor(all_stds, dtype=torch.float32).view(1, self.num_tissues, 1, 1).to(device)
        self.log_sigma = nn.Parameter(torch.log(init_stds))
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 1, H, W)
        B, _, H, W = x.shape
        expand_shape = (B, self.num_tissues, H, W)

        mu = self.mu.expand(expand_shape)
        log_sigma = self.log_sigma.expand(expand_shape)
        sigma = F.softplus(log_sigma) + 1e-6

        x_exp = x.expand(B, self.num_tissues, H, W)
        diff = x_exp - mu

        log_prob = -0.5 * (diff ** 2) / (sigma ** 2) - torch.log(sigma)
        gamma = F.softmax(log_prob, dim=1)  # (B, num_tissues, H, W)
        return gamma
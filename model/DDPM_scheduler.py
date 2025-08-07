import numpy as np
import torch

# schedulers for ddpm & ddim
def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def cosine_beta_schedule(timesteps: int, s=0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    a_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    a_cumprod = a_cumprod / a_cumprod[0]
    betas = 1 - (a_cumprod[1:] / a_cumprod[:-1])
    return torch.clip(betas, 0, 0.999).type(torch.float32)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1) -> torch.Tensor:
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    a_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    a_cumprod = a_cumprod / a_cumprod[0]
    betas = 1 - (a_cumprod[1:] / a_cumprod[:-1])
    return torch.clip(betas, 0, 0.999).type(torch.float32)
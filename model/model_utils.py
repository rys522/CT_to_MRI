import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from scipy import integrate
from typing import Callable

from torch import Tensor
import ot


device = torch.device("cuda:0")

def marginal_prob_std(t, sigma):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The standard deviation.
    """
    t = torch.tensor(t, device=device)
    return torch.sqrt((sigma ** (2 * t) - 1.0) / 2.0 / np.log(sigma)) #

def diffusion_coeff(t, sigma):
    """Compute the diffusion coefficient of our SDE.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The vector of diffusion coefficients.
    """
    return torch.tensor(sigma**t, device=device)

def gen_cat_input(
    x_t: Tensor,
    cond1: Tensor,
) -> tuple[Tensor, Tensor]:
    input = (x_t.type(torch.float32), cond1.type(torch.float32))
    return input


def set_sampler(sampler_type: str):
    if sampler_type == "ode":
        return ode_sampler
    elif sampler_type == "em":
        return Euler_Maruyama_sampler
    elif sampler_type == "pc":
        return pc_sampler
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")


class EMA(nn.Module):
    def __init__(self, model, decay=0.9999):
        super(EMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        # self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(
                self.module.state_dict().values(), model.state_dict().values()
            ):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(
            model, update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m
        )

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


def Euler_Maruyama_sampler(
    score_model :nn.Module,
    condition : Tensor,
    marginal_prob_std : Callable,
    diffusion_coeff : Callable,
    batch_size : int = 64,
    num_steps=500,
    device="cuda",
    eps=1e-3,
    image_size=(28, 28),
):
    """Generate samples from score-based models with the Euler-Maruyama solver.

    Args:
        score_model: A PyTorch model that represents the time-dependent score-based model.
        marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
        diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
        batch_size: The number of samplers to generate by calling this function once.
        num_steps: The number of sampling steps.
        Equivalent to the number of discretized time steps.
        device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
        eps: The smallest time step for numerical stability.

    Returns:
        Samples.
    """
    t = torch.ones(batch_size, device=device)
    init_x = (
        torch.randn(batch_size, 1, *image_size, device=device)
        * marginal_prob_std(t)[:, None, None, None]
    )
    time_steps = torch.linspace(1.0, eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
        for time_step in time_steps:
            input_tuple = gen_cat_input(x_t=x, cond1=condition)
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            mean_x = (
                x
                + (g**2)[:, None, None, None]
                * score_model(input_tuple, batch_time_step)
                * step_size
            )
            x = mean_x + torch.sqrt(step_size) * g[
                :, None, None, None
            ] * torch.randn_like(x)
    # Do not include any noise in the last sampling step.
    return mean_x.type(torch.float32)


def pc_sampler(
    score_model : nn.Module,
    condition : torch.Tensor,
    marginal_prob_std : Callable,
    diffusion_coeff : Callable,
    batch_size=32,
    snr=0.16,
    num_steps=500,
    device="cuda",
    eps=1e-3,
    image_size=(28, 28),
):
    """Generate samples from score-based models with Predictor-Corrector method.

    Args:
        score_model: A PyTorch model that represents the time-dependent score-based model.
        marginal_prob_std: A function that gives the standard deviation
        of the perturbation kernel.
        diffusion_coeff: A function that gives the diffusion coefficient
        of the SDE.
        batch_size: The number of samplers to generate by calling this function once.
        num_steps: The number of sampling steps.
        Equivalent to the number of discretized time steps.
        device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
        eps: The smallest time step for numerical stability.

    Returns:
        Samples.
    """
    # Step 1: define start time t=1 and random samples from prior data distribution
    t = torch.ones(batch_size, device=device)
    init_x = (
        torch.randn(batch_size, 1, *image_size, device=device)
        * marginal_prob_std(t)[:, None, None, None]
    )

    # Step 2: define reverse time grid and time intervals
    time_steps = np.linspace(1.0, eps, num_steps)
    step_size = time_steps[0] - time_steps[1]

    # Step 3: alternatively use Langevin sampling and reverse-time SDE with Euler approach to solve
    x = init_x
    with torch.no_grad():
        for time_step in time_steps:
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            input_tuple = gen_cat_input(x_t=x, cond1=condition)

            # Corrector step (Langevin MCMC)
            grad = score_model(input_tuple, batch_time_step)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm) ** 2

            for _ in range(10):
                x = (
                    x
                    + langevin_step_size * grad
                    + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)
                )
                input_tuple = gen_cat_input(x_t=x, cond1=condition)
                # Recompute the gradient after Langevin step

                grad = score_model(input_tuple, batch_time_step)
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                noise_norm = np.sqrt(np.prod(x.shape[1:]))
                langevin_step_size = 2 * (snr * noise_norm / grad_norm) ** 2

            # Predictor step (Euler-Maruyama)
            g = diffusion_coeff(batch_time_step)
            x_mean = (
                x
                + (g**2)[:, None, None, None]
                * score_model(input_tuple, batch_time_step)
                * step_size
            )
            x = x_mean + torch.sqrt(g**2 * step_size)[
                :, None, None, None
            ] * torch.randn_like(x)

        # The last step does not include any noise
    return x_mean.type(torch.float32)


def ode_sampler(
    score_model,
    condition,
    marginal_prob_std,
    diffusion_coeff,
    batch_size=64,
    atol=1e-5,
    rtol=1e-5,
    z=None,
    eps=1e-3,
    image_size=(28, 28),
):
    """Generate samplers from score-based models with ODE method"""

    # Step 1: define start time t=1 and initial x
    t = torch.ones(batch_size, device=device)
    if z is None:
        init_x = (
            torch.randn(batch_size, 1, *image_size, device=device)
            * marginal_prob_std(t)[:, None, None, None]
        )
    else:
        init_x = z
    shape = init_x.shape

    # Step 2: define score estimation function and ODE function
    def score_eval_wrapper(sample, time_steps):
        """A Wrapper of the score-based model for use by the ODE solver"""
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(
            time_steps, device=device, dtype=torch.float32
        ).reshape((sample.shape[0],))
        input_tuple = gen_cat_input(x_t=sample, cond1=condition)
        with torch.no_grad():
            score = score_model(input_tuple, time_steps)
        return score.cpu().numpy().reshape((-1,)).astype(np.float64)

    def ode_func(t, x):
        """The ODE function for use by the ODE solver"""
        time_steps = np.ones((shape[0],)) * t
        g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
        return -0.5 * (g**2) * score_eval_wrapper(x, time_steps)

    # Step 3: using ODE to solve value at t=eps
    res = integrate.solve_ivp(
        ode_func,
        (1.0, eps),
        init_x.reshape(-1).cpu().numpy(),
        rtol=rtol,
        atol=atol,
        method="RK45",
    )
    # print(f"Number of function evaluations: {res.nfev}")

    x = torch.from_numpy(res.y[:, -1].astype(np.float32)).to(device).reshape(shape)
    return x.type(torch.float32)


import numpy as np
import ot

import torch

def sinkhorn(
    source: torch.Tensor,   # [N, D]
    target: torch.Tensor,   # [N, D]
    epsilon: float = 0.1,   # regularization strength (2σ²)
    n_iters: int = 100,     # number of Sinkhorn iterations
) -> torch.Tensor:
    """
    Computes entropy-regularized OT plan T using Sinkhorn algorithm in PyTorch.

    Args:
        source: tensor of shape [N, D] (e.g., x1)
        target: tensor of shape [N, D] (e.g., x0)
        epsilon: entropy regularization (ε = 2σ²)
        n_iters: number of Sinkhorn iterations

    Returns:
        T: [N, N] OT plan matrix (soft matching)
    """
    N = source.size(0)
    device = source.device

    # Uniform distributions
    mu = torch.full((N,), 1.0 / N, device=device)
    nu = torch.full((N,), 1.0 / N, device=device)

    # Cost matrix: C_{ij} = ||x_i - y_j||^2
    cost = torch.cdist(source, target, p=2).pow(2)  # [N, N]

    # Initialize log dual potentials
    K = torch.exp(-cost / epsilon)  # kernel matrix
    u = torch.ones(N, device=device) / N
    v = torch.ones(N, device=device) / N

    # Sinkhorn iterations
    for _ in range(n_iters):
        u = mu / (K @ v + 1e-8)
        v = nu / (K.T @ u + 1e-8)

    # Transport plan π = diag(u) @ K @ diag(v)
    T = torch.diag(u) @ K @ torch.diag(v)

    T = T.clamp(min=1e-12)
    T = T / (T.sum(dim=1, keepdim=True) + 1e-8)

    return T  # shape [N, N]
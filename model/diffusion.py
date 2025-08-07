import itertools

import torch
import torch.nn as nn
from model.tunet import TimeUnet
from torch import Tensor
import functools
from common.logger import logger
from model.DDPM_scheduler import *
from model.model_utils import *
from model.CT_tissue import LearnableGMMDecomposer, mask_soft_tissue_only_tensor
from typing import Optional, Union
import math

from model.tunet_uncond import TimeUnet_uncond

from params import ModelConfig, GMM_ModelConfig

device = torch.device("cuda:0")

class Diffusion(nn.Module):
    def __init__(self, device: torch.device, modelconfig : ModelConfig):
        super().__init__()
        self.device = device
        self.timesteps: int = 1000
        self.sampler = set_sampler(modelconfig.sampler_type)


        self.u_network = TimeUnet(
            in_channels=modelconfig.unet_input_chan,
            out_channels=modelconfig.unet_output_chan,
            time_emb_dim=modelconfig.time_emb_dim,
            num_pool_layers=modelconfig.num_pool_layers,
            channels=modelconfig.unet_chans,
        )

        self.algorithm = modelconfig.algorithm
        if self.algorithm not in ["ddpm", "ddim", "flow", "score"]:
            raise NotImplementedError

        # Below for ddpm and ddim
        self.ddim_eta = modelconfig.ddim_eta
        self.beta_schedule = modelconfig.beta_schedule
        self.ddpm_target = modelconfig.ddpm_target

        # calc beta
        if modelconfig.beta_schedule == "cosine":
            self.betas = cosine_beta_schedule(timesteps=self.timesteps)
        elif modelconfig.beta_schedule == "linear":
            self.betas = linear_beta_schedule(timesteps=self.timesteps)
        elif modelconfig.beta_schedule == "sigmoid":
            self.betas = sigmoid_beta_schedule(timesteps=self.timesteps)
        else:
            raise KeyError("Unsupported beta schedule")
        self.betas = self.betas.to(device)

        # calc alphas
        self.alphas = 1.0 - self.betas
        self.a_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_a_cumprod = torch.sqrt(self.a_cumprod)
        self.sqrt_one_minus_a_cumprod = torch.sqrt(1 - self.a_cumprod)

        # Score EMA marginal prob std & diffusion coeff 

        self.ema_model = torch.nn.DataParallel(EMA(self.u_network, decay=0.9999)).to(device)

        self.sigma = modelconfig.sigma
        self.marginal_prob_std = functools.partial(marginal_prob_std, sigma=self.sigma)
        self.diffusion_coeff = functools.partial(diffusion_coeff, sigma=self.sigma)

        #flow
        self.sigma_min = modelconfig.sigma_min
        self.flow_type = modelconfig.flow_type

    ### training ###
    def forward(
        self,
        lab: Optional[Tensor] = None,
        cond: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        mode="train",
        
        interval = None
    ) ->  Union[Tensor, tuple[Tensor, Tensor]]:

        if mode == "recon":
            return self.recon(cond=cond, interval = interval) 
        if not (isinstance(lab, Tensor) and isinstance(cond, Tensor)):
            raise KeyError("All inputs must be tensors.")

        if not (lab.dim() == 4 and cond.dim() == 4):
            raise ValueError("All tensors must be 4D.")

        if self.algorithm in ["flow"]:
            return self.forward_flow(lab=lab, cond=cond, mask=mask)
        elif self.algorithm in ["ddpm", "ddim"]:
            return self.forward_ddpm_ddim(lab=lab, cond=cond)
        elif self.algorithm in ["score"]:
            return self.forward_score(lab=lab, cond = cond)
        else:
            raise NotImplementedError
        
    # DDPM_DDIM forward fn    
    def forward_ddpm_ddim(
        self,
        lab: Tensor,
        cond: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
    ]:
        batch_size = lab.shape[0]
        t = torch.randint(
            low=0,
            high=self.timesteps,
            size=(batch_size,),
            device=self.device,
            dtype=torch.long,
        )

        noise = torch.randn_like(lab).to(self.device)

        sqrt_a_cumprod = self.sqrt_a_cumprod.to(t.device)
        sqrt_one_minus_a_cumprod = self.sqrt_one_minus_a_cumprod.to(t.device)

        # making noised images x_t
        x_t = (
            sqrt_a_cumprod.gather(0, t).view(batch_size, 1, 1, 1) * lab
            + sqrt_one_minus_a_cumprod.gather(0, t).view(batch_size, 1, 1, 1) * noise
        ).type(torch.float32)

        input = gen_cat_input(x_t=x_t, cond1=cond) #concat input and condition

        output = self.u_network(
            input,
            t.type(torch.float32),
        )

        if self.ddpm_target == "noise":
            target = noise
        elif self.ddpm_target == "velocity":
            target = (
                sqrt_a_cumprod.gather(0, t).view(batch_size, 1, 1, 1) * noise
                - sqrt_one_minus_a_cumprod.gather(0, t).view(batch_size, 1, 1, 1) * lab
            )
        elif self.ddpm_target == "start":
            target = lab
        else:
            raise KeyError("Unsupported beta model_target")

        return (
            output,
            target,
        )
    
    ## Flow model forward fn
    def forward_flow(
        self,
        lab: Tensor,
        cond: Tensor,
        mask: Optional[Tensor] = None,
    ) -> tuple[
        Tensor,
        Tensor,
    ]:
        
        sigma_min = self.sigma_min 
        batch_size = lab.shape[0]
        t = torch.randint(
            low=0,
            high=self.timesteps,
            size=(batch_size,),
            device=self.device,
            dtype=torch.long,
        )

        noise = torch.randn_like(lab).to(self.device)

        t_n = t.float() / float(self.timesteps)
        if self.flow_type == "vp":
            x_t = (
                (1 -(1-sigma_min) * t_n).view(batch_size, 1, 1, 1) * noise + t_n.view(batch_size, 1, 1, 1) * lab
            ).type(torch.float32)
        elif self.flow_type == "ot":
            x_t = (
                (1 -(1-sigma_min) * t_n).view(batch_size, 1, 1, 1) * noise + t_n.view(batch_size, 1, 1, 1) * lab
            ).type(torch.float32)
        elif self.flow_type == "rect":
            x_t = (
                (1 - t_n).view(batch_size, 1, 1, 1) * noise + t_n.view(batch_size, 1, 1, 1) * lab
            ).type(torch.float32)
        else:
            raise KeyError("Unsupported beta model_target")

        input = gen_cat_input(x_t=x_t, cond1=cond)

        pred = self.u_network(
            input,
            t_n.type(torch.float32),
        )
        
        output = pred

        if self.flow_type == "vp":
            target = lab - (1-sigma_min) * noise
        elif self.flow_type == "ot":
            target = lab - (1-sigma_min) * noise
        elif self.flow_type == "rect":
            output = pred / (torch.norm(pred.view(batch_size, -1), dim=1, keepdim=True) + 1e-8).view(batch_size, 1, 1, 1)
            direction = lab - noise
            norm = torch.norm(direction.view(batch_size, -1), dim=1, keepdim=True) + 1e-8
            target = direction / norm.view(batch_size, 1, 1, 1)            
        else:
            raise KeyError("Unsupported beta model_target")

        return (
            output,
            target,
        )


    def forward_score(
        self, 
        lab: Tensor,
        cond: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
    ]:
        
        eps = 1e-5

        # Step 1: Sample random time t ~ Uniform(eps, 1)
        t = torch.rand(lab.size(0), device=lab.device) * (1.0 - eps) + eps  # shape (B,)

        # Step 2: Perturb x with Gaussian noise scaled by std(t)
        device = lab.device

        noise = torch.randn_like(lab).to(device)
        std = self.marginal_prob_std(t).view(-1, 1, 1, 1).to(device)
        perturbed_x = lab + std * noise

        input_tuple = gen_cat_input(x_t=perturbed_x, cond1=cond)
        score = self.u_network(input_tuple, t)
        output = score * std[:, None, None, None]

        return (
            output, 
            -noise,
        )

    ### inference ###
    @torch.inference_mode()
    def recon(self, *args, **kwargs) -> Tensor:
        """
        Inference method for diffusion models.

        Args:
            *args:
                - out_size (torch.Size): Output tensor size.
            **kwargs:
                - interval (float, optional): Interval parameter (default: 50).

        Returns:
            Tensor: Reconstructed tensor.

        Raises:
            NotImplementedError: If the specified algorithm is not supported.
        """

        if self.algorithm in ["flow"]:
            return self.recon_flow(*args, **kwargs)

        elif self.algorithm in ["ddpm", "ddim"]:
            return self.recon_ddpm_ddim(*args, **kwargs)
        elif self.algorithm in ["score"]:
            return self.recon_score(*args, **kwargs)
        else:
            raise NotImplementedError
    
    @torch.inference_mode()
    def recon_flow(
        self,
        cond: Tensor,
        interval: int = 50,
    ) -> Tensor:
        device = cond.device

        batch_size = cond.shape[0]
        ddim_steps = int(self.timesteps / interval)
        times = torch.linspace(0, 1, ddim_steps, dtype=torch.float32, device=device)

        dt = interval / self.timesteps

        noise = torch.randn_like(cond).to(self.device)
        x_t = noise.type(torch.float32)

        for time in times:
            logger.trace(f"Diffusion time : {time}")
            t_batch = torch.full(
                size=(batch_size,),
                fill_value=time.item() if isinstance(time, torch.Tensor) else time,
                device=self.device,
                dtype=torch.float32,
            )

            if torch.isnan(t_batch).any() or torch.isinf(t_batch).any():
                logger.error(f"[recon_flow] t_batch contains NaN or Inf: {t_batch}")
            input = gen_cat_input(x_t=x_t, cond1=cond)

            pred = self.u_network(
                input,
                t_batch.type(torch.float32),
            )

            assert isinstance(pred, Tensor), "predicted_noise should be a Tensor"

            if torch.isnan(pred).any():
                logger.error(f"[recon_flow] NaN detected in pred at t={time}")
            if torch.isinf(pred).any():
                logger.error(f"[recon_flow] Inf detected in pred at t={time}")
            logger.debug(f"[recon_flow] pred stats | min: {pred.min().item():.4f}, max: {pred.max().item():.4f}")


            if self.flow_type == "rect":
                pred = pred / (pred.view(batch_size, -1).norm(dim=1, keepdim=True) + 1e-8).view(batch_size, 1, 1, 1)

            x_t = x_t + dt * pred

        return x_t.type(torch.float32)
        
    @torch.inference_mode()
    def predict_noise(
        self,
        network_output: torch.Tensor,
        a_cumprod_t: torch.Tensor,
        x_t: torch.Tensor,
        pred_x0: torch.Tensor,
    ) -> torch.Tensor:
        if self.ddpm_target == "noise":
            predicted_noise = network_output
        elif self.ddpm_target in ["velocity", "start"]:
            predicted_noise = (torch.sqrt(1.0 / a_cumprod_t) * x_t - pred_x0) / torch.sqrt(
                1.0 / a_cumprod_t - 1
            )
        else:
            raise KeyError("Unsupported beta model_target")
        return predicted_noise

    @torch.inference_mode()
    def predict_x0(
        self,
        x_t: torch.Tensor,
        a_cumprod_t: torch.Tensor,
        network_output: torch.Tensor,
    ) -> torch.Tensor:
        if self.ddpm_target == "noise":
            pred_x0 = (x_t - torch.sqrt(1.0 - a_cumprod_t) * network_output) / torch.sqrt(
                a_cumprod_t
            )
        elif self.ddpm_target == "velocity":
            pred_x0 = torch.sqrt(a_cumprod_t) * x_t - torch.sqrt(1 - a_cumprod_t) * network_output
        elif self.ddpm_target == "start":
            pred_x0 = network_output
        else:
            raise KeyError("Unsupported beta model_target")

        return pred_x0

    @torch.inference_mode()
    def recon_ddpm_ddim(
        self,
        cond: Tensor,
        interval: float = 50,
    ) -> Tensor:
        batch_size = cond.shape[0]
        ddim_steps = int(self.timesteps / interval)

        times = torch.linspace(0, self.timesteps - 1, ddim_steps, dtype=torch.long)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(itertools.pairwise(times))

        noise = torch.randn_like(cond).to(self.device)
        x_t = noise.type(torch.float32)

        for time, time_next in time_pairs:
            # set hyper
            t_batch = torch.full(
                (batch_size,),
                time,
                device=self.device,
                dtype=torch.long,
            )

            t_next_batch = torch.full(
                (batch_size,),
                time_next,
                device=self.device,
                dtype=torch.long,
            )

            input = gen_cat_input(x_t=x_t, cond1=cond)

            # model inference
            network_output = self.u_network(
                input,
                t_batch.type(torch.float32),
            )
            assert isinstance(network_output, torch.Tensor), "predicted_noise should be a Tensor"

            a_cumprod = self.a_cumprod.to(self.device)
            a_cumprod_t = a_cumprod.gather(0, t_batch).view(batch_size, 1, 1, 1)
            a_cumpord_t_next = a_cumprod.gather(0, t_next_batch).view(batch_size, 1, 1, 1)

            ## calc predx0
            pred_x0 = self.predict_x0(
                x_t=x_t,
                a_cumprod_t=a_cumprod_t,
                network_output=network_output,
            )

            ## recon
            predicted_noise = self.predict_noise(
                network_output=network_output,
                a_cumprod_t=a_cumprod_t,
                x_t=x_t,
                pred_x0=pred_x0,
            )

            # DDIM
            if self.algorithm == "ddim":
                alpha_t = self.alphas.gather(0, t_batch).view(batch_size, 1, 1, 1)
                noise = torch.randn_like(x_t)
                variance = self.ddim_eta * torch.sqrt(
                    (1 - alpha_t) * ((1 - a_cumpord_t_next) / (1 - a_cumprod_t))
                )
                x_next = (
                    torch.sqrt(a_cumpord_t_next) * pred_x0
                    + torch.sqrt(1 - a_cumpord_t_next - variance**2) * predicted_noise
                    + variance * noise
                )

            # DDPM
            elif self.algorithm == "ddpm":
                alpha_t = self.alphas.gather(0, t_batch).view(batch_size, 1, 1, 1)
                noise = torch.randn_like(x_t)
                variance = 1 * torch.sqrt(
                    (1 - alpha_t) * ((1 - a_cumpord_t_next) / (1 - a_cumprod_t))
                )
                x_next = (
                    x_t / torch.sqrt(alpha_t)
                    - (1 - alpha_t) / torch.sqrt((alpha_t) * (1 - a_cumprod_t)) * predicted_noise
                    + variance * noise
                )
            else:
                raise NotImplementedError

            x_t = x_next

        return x_t.type(torch.float32)
    

    @torch.inference_mode()
    def recon_score(
        self,
        cond: Tensor,
        interval: float = 50,
    ) -> Tensor:
        output = self.sampler(
            u_network=self.u_network,
            condition=cond,
            marginal_prob_std=self.marginal_prob_std,
            diffusion_coeff=self.diffusion_coeff,
            batch_size=cond.shape[0],
            image_size=cond.shape[2:], 
        )

        return output

class SBFlow(nn.Module):
    def __init__(
        self, 
        device: torch.device, 
        modelconfig: ModelConfig,
    ):
        super().__init__()
        self.device = device
        self.timesteps: int = 1000
        if modelconfig.unet_input_chan == 1:
            self.u_network = TimeUnet_uncond(
                in_chans=modelconfig.unet_input_chan,
                out_chans=modelconfig.unet_output_chan,
                time_emb_dim=modelconfig.time_emb_dim,
                num_pool_layers=modelconfig.num_pool_layers,
                chans=modelconfig.unet_chans,
            )
        else:
            self.u_network = TimeUnet(
                in_channels=modelconfig.unet_input_chan,
                out_channels=modelconfig.unet_output_chan,
                time_emb_dim=modelconfig.time_emb_dim,
                num_pool_layers=modelconfig.num_pool_layers,
                channels=modelconfig.unet_chans,
            )

        self.chans = modelconfig.unet_input_chan
        self.sigma = modelconfig.sigma_sb
        self.sigma_min = modelconfig.sigma_min

    def sample_ot_pairs(self, noise_batch, mri_batch, sigma, method="soft"):
        """
        Sample (x0, x1) pairs from OT plan between noise and MRI.

        Args:
            noise_batch: [B, 1, H, W] noise samples ~ N(0, I)
            mri_batch:   [B, 1, H, W] target MRI images
            sigma:       float, used in entropic regularization (ε = 2σ²)
            method:      "soft" or "hard"

        Returns:
            x0: noise samples matched to MRI (i.e., reordered)
            x1: MRI samples (unchanged)
        """
        B = noise_batch.size(0)

        # Flatten to [B, D] for cost computation
        noise_flat = noise_batch.view(B, -1)  # [B, D]
        mri_flat   = mri_batch.view(B, -1)    # [B, D]

        # OT plan: π[i, j] = P(MRI[i] ↔ noise[j])
        with torch.no_grad():
            pi = sinkhorn(mri_flat, noise_flat, epsilon=2 * sigma**2)  # [B, B]

        # Sample index of noise[j] per each MRI[i]
        if method == "soft":
            indices_x0 = torch.multinomial(pi, num_samples=1).squeeze(1)  # [B]
        else:
            indices_x0 = pi.argmax(dim=1)  # [B]

        # Matched noise ↔ MRI
        x0 = noise_batch[indices_x0]  # [B, 1, H, W]

        return x0

    def forward(
        self,
        lab: Optional[Tensor] = None,    # [B, 1, H, W] MRI (target)
        cond: Optional[Tensor] = None,   # [B, 1, H, W] CT (conditioning)
        tissue: Optional[Tensor] = None,
        mode="train",
        interval=None
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        
        if mode == "recon":
            return self.recon(cond=cond, interval=interval)

        assert lab is not None and cond is not None
        B, _, H, W = lab.shape
        sigma = self.sigma

        # 1. Sample x0, x1 from OT-matched pairs
        x1 = lab
        x0 = cond

        # 2. Sample t ∈ (0,1)
        t = torch.rand(B, device=self.device)  # [B]
        t_broadcast = t.view(B, 1, 1, 1)

        # 3. Brownian bridge x ~ N(μ_t, σ² t(1−t))
        mu_t = t_broadcast * x1 + (1 - t_broadcast) * x0
        std_t = (sigma * (t * (1 - t)).sqrt()).view(B, 1, 1, 1)
        x = mu_t + std_t * torch.randn_like(mu_t)

        # 4. Compute u_t(x | x0, x1)
        numer = (1 - 2 * t_broadcast) * (x - mu_t)
        denom = 2 * t_broadcast * (1 - t_broadcast) + 1e-8
        u_t = numer / denom + (x1 - x0)  # [B, 1, H, W]

        # 5. Predict vector field v_θ(x, t, cond=CT)
        if self.chans >1:
            input = gen_cat_input(x_t=x, cond1=cond)  # cond aligns with x1
        else:
            input  = x

        pred = self.u_network(
            input,
            t.type(torch.float32),  # [B]
        )

        return pred, u_t


    @torch.inference_mode()
    def recon(
        self,
        cond: Tensor,
        interval: int = 50,
    ) -> Tensor:
        device = cond.device

        batch_size = cond.shape[0]
        ddim_steps = int(self.timesteps / interval)
        times = torch.linspace(0, 1, ddim_steps, dtype=torch.float32, device=device)

        dt = interval / self.timesteps

        x_t = cond

        for time in times:
            logger.trace(f"Diffusion time : {time}")
            t_batch = torch.full(
                size=(batch_size,),
                fill_value=time.item() if isinstance(time, torch.Tensor) else time,
                device=self.device,
                dtype=torch.float32,
            )
            if self.chans >1:
                input = gen_cat_input(x_t=x_t, cond1=cond)
            else:
                input = x_t

            pred = self.u_network(
                input,
                t_batch.type(torch.float32),
            )

            assert isinstance(pred, Tensor), "predicted_noise should be a Tensor"
            x_t = x_t + dt * pred

        return x_t.type(torch.float32)
    


class HybridDiffusion(nn.Module):
    def __init__(
        self, 
        device: torch.device, 
        modelconfig: ModelConfig,
        gmm_modelconfig: GMM_ModelConfig,
    ):
        super().__init__()
        self.device = device
        self.timesteps: int = 1000

        self.u_network = TimeUnet(
            in_channels=modelconfig.unet_input_chan,
            out_channels=modelconfig.unet_output_chan,
            time_emb_dim=modelconfig.time_emb_dim,
            num_pool_layers=modelconfig.num_pool_layers,
            channels=modelconfig.unet_chans,
        )

        self.gmm_model = LearnableGMMDecomposer(device=device, modelconfig=gmm_modelconfig)
        self.gmm_modelconfig = gmm_modelconfig

        self.sigma = modelconfig.sigma_sb
        self.sigma_min = modelconfig.sigma_min

    def sample_ot_pairs(self, noise_batch, mri_batch, sigma, method="soft"):
        """
        Sample (x0, x1) pairs from OT plan between noise and MRI.

        Args:
            noise_batch: [B, 1, H, W] noise samples ~ N(0, I)
            mri_batch:   [B, 1, H, W] target MRI images
            sigma:       float, used in entropic regularization (ε = 2σ²)
            method:      "soft" or "hard"

        Returns:
            x0: noise samples matched to MRI (i.e., reordered)
            x1: MRI samples (unchanged)
        """
        B = noise_batch.size(0)

        # Flatten to [B, D] for cost computation
        noise_flat = noise_batch.view(B, -1)  # [B, D]
        mri_flat   = mri_batch.view(B, -1)    # [B, D]

        # OT plan: π[i, j] = P(MRI[i] ↔ noise[j])
        with torch.no_grad():
            pi = sinkhorn(mri_flat, noise_flat, epsilon=2 * sigma**2)  # [B, B]

        # Sample index of noise[j] per each MRI[i]
        if method == "soft":
            indices_x0 = torch.multinomial(pi, num_samples=1).squeeze(1)  # [B]
        else:
            indices_x0 = pi.argmax(dim=1)  # [B]

        # Matched noise ↔ MRI
        x0 = noise_batch[indices_x0]  # [B, 1, H, W]

        return x0

    def forward(
        self,
        lab: torch.Tensor = None,          # Target MRI
        cond: torch.Tensor = None,         # [B, 1, H, W] CT input
        tissue: torch.Tensor = None,       # [B, 3, H, W] GT tissue map
        mode: str = "train",
        interval=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        device = cond.device

        ################   tissue   ##################
        # Get tissue probability map from GMM
        ct_tissue = self.gmm_model(mask_soft_tissue_only_tensor(cond))

        # Concatenate CT and tissue information according to lambda


        # Extract FAST-based tissue probability (CSF, GM, WM) and normalize
        tissue_order = list(self.gmm_modelconfig.tissue_dict.keys())
        tissue_indices_fast = [
            tissue_order.index("csf"),
            tissue_order.index("gray_matter"),
            tissue_order.index("white_matter"),
        ]
        ct_tissue_fast = ct_tissue[:, tissue_indices_fast]            # [B, 3, H, W]
        ct_tissue_fast_sum = ct_tissue_fast.sum(dim=1, keepdim=True)  # [B, 1, H, W]
        ct_tissue_fast_norm = ct_tissue_fast / (ct_tissue_fast_sum + 1e-8)

        ct_tissue_fast_norm = ct_tissue_fast_norm.to(device)
        output_tissue = ct_tissue_fast_norm

        if mode == "recon":
            return self.recon(cond=cond, interval=interval , tissue = output_tissue)
        
        if not (isinstance(lab, Tensor) and isinstance(cond, Tensor)):
            raise KeyError("All inputs must be tensors.")

        if not (lab.dim() == 4 and cond.dim() == 4):
            raise ValueError("All tensors must be 4D.")
        

        ############# sb flow ################
        sigma_min = self.sigma_min

        x1 = lab        # [B, 1, H, W] MRI (target)
        ct = cond       # [B, 1, H, W] CT (conditioning)

        if mode == "recon":
            # For reconstruction, usually inference doesn't require x0, just use ct as cond
            return self.recon(cond=ct, interval=interval)

        # Check input
        if not (isinstance(x1, Tensor) and isinstance(ct, Tensor)):
            raise KeyError("All inputs must be tensors.")
        if not (x1.dim() == 4 and ct.dim() == 4):
            raise ValueError("All tensors must be 4D.")

        B, _, H, W = x1.shape

        # 2. OT match ct ↔ MRI
        x0 = cond
        # Now (x0, x1_matched) is the SB-CFM pair

        # 3. Sample random t ∈ (0, 1)        
        t = torch.randint(
            low=0,
            high=self.timesteps,
            size=(B,),
            device=self.device,
            dtype=torch.long,
        )

        t_n = t / self.timesteps

        # 3. Brownian bridge x ~ N(μ_t, σ² t(1−t))

        x_t = (
            (1 -(1-sigma_min) * t_n).view(B, 1, 1, 1) * x0 + t_n.view(B, 1, 1, 1) * x1
            ).type(torch.float32)

        # 12. Predict vector field v_θ(x, t, cond=CT)
        input = gen_cat_input(x_t=x_t, cond1 = tissue)

        pred = self.u_network(
            input,
            t.type(torch.float32),
        )   # [B,1,H,W]

        # --- Network prediction ---
        output_sb = pred
        
        target_sb = lab - (1-sigma_min) * cond


        return output_sb, target_sb , output_tissue , tissue


    @torch.inference_mode()
    def recon(
        self,
        cond: Tensor,
        tissue: Tensor,
        interval: int = 50,

    ) -> Tensor:
        
        device = cond.device

        batch_size = cond.shape[0]
        ddim_steps = int(self.timesteps / interval)
        times = torch.linspace(0, 1, ddim_steps, dtype=torch.float32, device=device)

        dt = interval / self.timesteps

        x_t = cond

        for time in times:
            logger.trace(f"Diffusion time : {time}")
            t_batch = torch.full(
                size=(batch_size,),
                fill_value=time.item() if isinstance(time, torch.Tensor) else time,
                device=self.device,
                dtype=torch.float32,
            )

            input = gen_cat_input(x_t=x_t, cond1 = tissue)

            pred = self.u_network(
                input,
                t_batch.type(torch.float32),
            )

            x_t = x_t + dt * pred

        return x_t.type(torch.float32)


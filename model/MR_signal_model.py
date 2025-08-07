import torch
import torch.nn as nn

from params import MRSignal_ModelConfig

class MRSignalModel(nn.Module):
    def __init__(self, device: torch.device, modelconfig: MRSignal_ModelConfig) -> None:
        super().__init__()
        self.device = device

        self.sequence = modelconfig.sequence
        self.tesla = modelconfig.tesla

        self.TR = torch.tensor(modelconfig.TR / 1000.0, device=device)  # seconds
        self.TI = torch.tensor(modelconfig.TI / 1000.0, device=device)
        #self.TE = torch.tensor(modelconfig.TE / 1000.0, device=device)
        #self.modality = modelconfig.modality

    def forward(self, log_q_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            log_q_map: [B, 3, H, W] → log(PD), log(T1), log(T2)
            log_scanner_gain: [B, 1, 1, 1] or broadcastable

        Returns:
            signal: [B, 1, H, W] → synthetic T1-weighted MRI
        """
        log_pd = log_q_map[:, 0:1, :, :]  # [B, 1, H, W]
        """
        log_t1_t2 = torch.exp(torch.clamp(log_q_map[:, 1:3, :, :], -20, 30))  # [B, 2, H, W]
        t1 = log_t1_t2[:, 0:1, :, :]
        t2 = log_t1_t2[:, 1:2, :, :]
        """

        t1 = torch.exp(torch.clamp(log_q_map[:, 1:2, :, :], -20, 30))

        # Broadcasting fixed TE, TR, TI
        #TE = self.TE.view(1, 1, 1, 1)
        TR = self.TR.view(1, 1, 1, 1)
        TI = self.TI.view(1, 1, 1, 1)

        # MPRAGE signal model (T1-weighted)
        if "mpr" in self.sequence:
            # MPRAGE
            signal = log_pd * (1 - (2 * torch.exp(-TI / t1)) / (1 + torch.exp(-TR / t1)))
        elif "tfe" in self.sequence:
            # TFE
            signal = log_pd * (1 - torch.exp(-TR / t1))
        else:
            # fallback
            signal = log_pd * (1 - torch.exp(-TR / t1))

        #t2_se = torch.exp(log_pd) * (1 -  torch.exp(-TR / t1)) * torch.exp(-TE / t2)
        #lair = torch.exp(log_pd) * (1 - (2 * torch.exp(-TI / t1)) + torch.exp(-TR / t1)) * torch.exp(-TE / t2)
        """
        if self.modality in ["T1w"]:
            signal = t1w_mprage
        elif self.modality in ["T2w"]:
            signal = t2_se
        elif self.modality in ["Flair"]:
            signal = flair
        elif self.modality in ["PD"]:
            signal = log_pd
        """
        return signal  # [B, 1, H, W]
    


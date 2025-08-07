import torch
from torch import nn
import torch.nn.functional as F

# SSIM Loss Implementation
class SSIMLoss(nn.Module):
    def __init__(
        self,
        win_size: int = 7,
        k1: float = 0.01,
        k2: float = 0.03,
        reduced: bool = True,
    ):
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size**2)
        NP = win_size**2
        self.cov_norm = NP / (NP - 1)

        self.reduced = reduced
    
    def _ssim_channel(self, X, Y, Mask=None):
        data_range = 1
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        w = self.w.to(X.device)

        ux = F.conv2d(X, w)
        uy = F.conv2d(Y, w)
        uxx = F.conv2d(X * X, w)
        uyy = F.conv2d(Y * Y, w)
        uxy = F.conv2d(X * Y, w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux**2 + uy**2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        if Mask is not None:
            S = S * Mask[:, :, 3:-3, 3:-3]

        return S


    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        Mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Multi-channel SSIM loss: returns (1 - mean SSIM) averaged over all channels.
        X, Y: [B, C, H, W]
        """
        assert X.shape == Y.shape
        C = X.shape[1]
        ssim_list = []
        for c in range(C):
            S = self._ssim_channel(X[:, c:c+1], Y[:, c:c+1], Mask)
            if self.reduced:
                ssim_list.append(S.mean())
            else:
                ssim_list.append(S)
        ssim_val = torch.stack(ssim_list, dim=0).mean()
        return 1 - ssim_val


# gradient loss for image reconstruction
def grad_loss(x, y, device="cuda"):
    mean = 0
    cx = [[[[1, -1]]]]
    cy = [[[[1], [-1]]]]
    cx = torch.FloatTensor(cx).to(device=device, dtype=torch.float32)
    cy = torch.FloatTensor(cy).to(device=device, dtype=torch.float32)
    for i in range(x.shape[1]):
        x1 = x[:, i : i + 1, :, :]
        y1 = y[:, i : i + 1, :, :]
        xx = F.conv2d(x1, cx, padding=1)
        xy = F.conv2d(x1, cy, padding=1)
        yx = F.conv2d(y1, cx, padding=1)
        yy = F.conv2d(y1, cy, padding=1)
        mean += 0.5 * (torch.mean(torch.abs(xx - yx)) + torch.mean(torch.abs(xy - yy)))
    return mean

import math

import torch
from torch import nn, Tensor
from common.logger import logger


class TimeEmbedding(nn.Module):
    """
    A class to handle time embeddings for a given time step.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        """
        Returns a list of embeddings based on the time step.
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


class TimeConvAttentionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
    ):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_channels * 2),
            nn.SiLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
        )

        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 8, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels // 8, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
        )

        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.skip_gate = nn.Sequential(nn.Conv2d(in_channels, 1, kernel_size=1), nn.Sigmoid())
        self.output_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """
        Forward pass through the TimeConvAttentionBlock.
        """
        identity = self.skip(x)
        gate = self.skip_gate(x)
        identity = identity * gate

        x = self.layer1(x)

        if torch.isnan(t_emb).any():
            logger.error("NaN in input(t_emb)")
        logger.debug(f"t_emb stats | min: {t_emb.min().item():.4f}, max: {t_emb.max().item():.4f}, mean: {t_emb.mean().item():.4f}")
        t_emb = self.time_mlp(t_emb)
        if torch.isnan(t_emb).any():
            logger.error("NaN in time_mlp(t_emb)")
        t_emb = t_emb[:, :, None, None]
        shift, bias = t_emb.chunk(2, dim=1)

        if torch.isnan(shift).any(): logger.error("NaN in shift")
        if torch.isnan(bias).any(): logger.error("NaN in bias")

        x = x * (1 + shift) + bias

        x = self.layer2(x)
        x = x * self.se_block(x)

        x = self.layer3(x)
        return identity + self.output_scale * x


class FirstLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        time_emb_dim: int,
    ) -> None:
        super().__init__()
        self.x_t = TimeConvAttentionBlock(1, channels // 2, time_emb_dim)
        self.in_conv = TimeConvAttentionBlock(in_channels - 1, channels // 2 , time_emb_dim)

        self.final_conv = TimeConvAttentionBlock(
            channels,
            channels,
            time_emb_dim,
        )

    def forward(
        self,
        x: tuple[Tensor, Tensor],
        t_emb: Tensor,
    ) -> Tensor:
        """
        Forward pass through the FirstLayer.
        """
        x_t = x[0]
        cond = x[1] #[B, N ,H, W]

        x_t_out = self.x_t(x_t, t_emb)
        cond1_out = self.in_conv(cond, t_emb)
        output = torch.cat([x_t_out, cond1_out], dim=1)

        output = self.final_conv(output, t_emb)
        return output


class FinalLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
    ):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.InstanceNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, in_channels * 2),
            nn.SiLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.InstanceNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.skip_gate = nn.Sequential(nn.Conv2d(in_channels, 1, kernel_size=1), nn.Sigmoid())
        self.output_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """
        Forward pass through the FinalLayer.
        """
        identity = self.skip(x)
        gate = self.skip_gate(x)
        identity = identity * gate

        x = self.layer1(x)

        t_emb = self.time_mlp(t_emb)[:, :, None, None]
        shift, bias = t_emb.chunk(2, dim=1)
        x = x * (1 + shift) + bias

        x = self.layer2(x)

        return identity + self.output_scale * x


class TimeUnet(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        time_emb_dim: int = 256,
        num_pool_layers: int = 5,
        channels: int = 32,
    ):
        super().__init__()

        self.output_channels = out_channels
        self.num_pool_layers = num_pool_layers

        self.time_mlp = TimeEmbedding(time_emb_dim)

        self.first_layer = FirstLayer(
            in_channels,
            channels,
            time_emb_dim,
        )

        self.down_conv_layers = self.create_down_conv_layers(
            channels,
            num_pool_layers,
        )

        self.down_sample_layers = self.create_down_sample_layers(
            channels,
            time_emb_dim,
            num_pool_layers,
        )

        # Bottleneck layer
        self.bottleneck_conv = TimeConvAttentionBlock(
            channels * (2 ** (num_pool_layers - 1)),
            channels * (2 ** (num_pool_layers - 1)),
            time_emb_dim,
        )

        self.up_conv_layers = self.create_up_conv_layers(
            channels,
            num_pool_layers,
        )

        self.up_sample_layers = self.create_up_sample_layers(
            channels,
            time_emb_dim,
            num_pool_layers,
        )

        self.concat_conv = TimeConvAttentionBlock(
            channels + in_channels,
            channels,
            time_emb_dim,
        )

        self.final_conv = FinalLayer(
            in_channels=channels,
            out_channels=out_channels,
            time_emb_dim=time_emb_dim,
        )

    def create_down_conv_layers(self, channels: int, num_pool_layers: int):
        layers = nn.ModuleList()
        ch = channels
        for _ in range(num_pool_layers - 1):
            layers.append(nn.Conv2d(ch * 2, ch * 2, kernel_size=4, stride=2, padding=1))
            ch *= 2
        return layers

    def create_down_sample_layers(
        self, channels: int, time_emb_dim: int, num_pool_layers: int
    ):
        layers = nn.ModuleList()
        ch = channels
        for _ in range(num_pool_layers - 1):
            layers.append(TimeConvAttentionBlock(ch, ch * 2, time_emb_dim))
            ch *= 2
        return layers

    def create_up_conv_layers(self, channels: int, num_pool_layers: int):
        layers = nn.ModuleList()
        ch = channels * (2 ** (num_pool_layers - 1))
        for _ in range(num_pool_layers - 1):
            layers.append(nn.ConvTranspose2d(ch, ch, kernel_size=2, stride=2))
            ch //= 2
        layers.append(nn.Identity())
        return layers

    def create_up_sample_layers(
        self, channels: int, time_emb_dim: int, num_pool_layers: int
    ):
        layers = nn.ModuleList()
        ch = channels * (2 ** (num_pool_layers - 1))
        for _ in range(num_pool_layers - 1):
            layers.append(TimeConvAttentionBlock(ch * 2, ch // 2, time_emb_dim))
            ch //= 2
        layers.append(TimeConvAttentionBlock(ch * 2, ch, time_emb_dim))
        return layers

    def forward(
        self, x: tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor],
        t: Tensor,
    ) -> Tensor:
        t_emb = self.time_mlp(t)

        stack = []
        output = self.first_layer(x, t_emb)
        stack.append(output)
        for down_conv, layer in zip(
            self.down_conv_layers, self.down_sample_layers, strict=False
        ):
            output = layer(output, t_emb)
            stack.append(output)
            output = down_conv(output)

        output = self.bottleneck_conv(output, t_emb)

        for up_conv, layer in zip(
            self.up_conv_layers, self.up_sample_layers, strict=False
        ):
            downsampled_output = stack.pop()
            output = up_conv(output)

            B, C, W, H = downsampled_output.shape
            output = output[:, :, :W, :H]

            output = torch.cat([output, downsampled_output], dim=1)
            output = layer(output, t_emb)

        if isinstance(x, tuple):
            x = torch.cat(x, dim=1)
        output = self.concat_conv(torch.cat([output, x], dim=1), t_emb)
        output = self.final_conv(output, t_emb)
        return output

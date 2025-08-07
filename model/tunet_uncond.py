import math

import torch
from torch import Tensor, nn


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


# ConvBlock modified to accept time embeddings
class TimeConvAttentionBlock(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        time_emb_dim: int,
    ):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_chans),
            nn.SiLU(inplace=True),
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_chans * 2),
            nn.SiLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_chans),
            nn.SiLU(inplace=True),
        )

        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_chans, out_chans // 8, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_chans // 8, out_chans, kernel_size=1),
            nn.Sigmoid(),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_chans),
            nn.SiLU(inplace=True),
        )

        self.skip = (
            nn.Conv2d(in_chans, out_chans, kernel_size=1)
            if in_chans != out_chans
            else nn.Identity()
        )

        self.skip_gate = nn.Sequential(nn.Conv2d(in_chans, 1, kernel_size=1), nn.Sigmoid())
        self.output_scale = nn.Parameter(torch.ones(1))

    def forward(
        self,
        x: Tensor,
        t_emb: Tensor,
    ) -> Tensor:
        identity = self.skip(x)
        gate = self.skip_gate(x)
        identity = identity * gate

        x = self.layer1(x)

        t_emb = self.time_mlp(t_emb)[:, :, None, None]
        shift, bias = t_emb.chunk(2, dim=1)
        x = x * (1 + shift) + bias

        x = self.layer2(x)

        se_weight = self.se_block(x)
        x = x * se_weight

        x = self.layer3(x)

        return identity + self.output_scale * x


class FirstLayer(nn.Module):
    def __init__(
        self,
        in_chans: int,
        chans: int,
        time_emb_dim: int,
    ) -> None:
        super().__init__()

        self.final_conv = TimeConvAttentionBlock(
            in_chans,
            chans,
            time_emb_dim,
        )

    def forward(
        self,
        x: Tensor,
        t_emb: Tensor,
    ) -> Tensor:
        output = self.final_conv(x, t_emb)
        return output


class FinalLayer(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        time_emb_dim: int,
    ):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_chans, in_chans, kernel_size=1),
            nn.InstanceNorm2d(out_chans),
            nn.SiLU(inplace=True),
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, in_chans * 2),
            nn.SiLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=1),
            nn.InstanceNorm2d(out_chans),
            nn.SiLU(inplace=True),
        )

        self.skip = (
            nn.Conv2d(in_chans, out_chans, kernel_size=1)
            if in_chans != out_chans
            else nn.Identity()
        )

        self.skip_gate = nn.Sequential(nn.Conv2d(in_chans, 1, kernel_size=1), nn.Sigmoid())
        self.output_scale = nn.Parameter(torch.ones(1))

    def forward(
        self,
        x: Tensor,
        t_emb: Tensor,
    ) -> Tensor:
        identity = self.skip(x)
        gate = self.skip_gate(x)
        identity = identity * gate

        x = self.layer1(x)

        t_emb = self.time_mlp(t_emb)[:, :, None, None]
        shift, bias = t_emb.chunk(2, dim=1)
        x = x * (1 + shift) + bias

        x = self.layer2(x)

        return identity + self.output_scale * x


# UNet model with time embedding
class TimeUnet_uncond(nn.Module):
    def __init__(
        self,
        in_chans: int = 2,
        out_chans: int = 2,
        chans: int = 32,
        num_pool_layers: int = 5,
        time_emb_dim: int = 256,
    ):
        super().__init__()

        self.out_chans = out_chans
        self.num_pool_layers = num_pool_layers

        # Time embedding
        self.time_mlp = TimeEmbedding(time_emb_dim)

        self.first_layer = FirstLayer(
            in_chans,
            chans,
            time_emb_dim,
        )

        self.down_conv_layers = self.create_down_conv_layers(
            chans,
            num_pool_layers,
        )

        # Down-sampling layers
        self.down_sample_layers = self.create_down_sample_layers(
            chans,
            num_pool_layers,
            time_emb_dim,
        )

        # Bottleneck layer
        self.bottleneck_conv = TimeConvAttentionBlock(
            chans * (2 ** (num_pool_layers - 1)),
            chans * (2 ** (num_pool_layers - 1)),
            time_emb_dim,
        )

        self.up_conv_layers = self.create_up_conv_layers(
            chans,
            num_pool_layers,
        )

        # Up-sampling layers
        self.up_sample_layers = self.create_up_sample_layers(
            chans,
            num_pool_layers,
            time_emb_dim,
        )

        self.concat_conv = TimeConvAttentionBlock(
            chans + in_chans,
            chans,
            time_emb_dim,
        )

        self.final_conv = FinalLayer(
            chans,
            out_chans,
            time_emb_dim,
        )

    def create_down_conv_layers(
        self,
        chans: int,
        num_pool_layers: int,
    ):
        layers = nn.ModuleList()
        ch = chans
        for _ in range(num_pool_layers - 1):
            layers.append(nn.Conv2d(ch * 2, ch * 2, kernel_size=4, stride=2, padding=1))
            ch *= 2
        return layers

    def create_down_sample_layers(
        self,
        chans: int,
        num_pool_layers: int,
        time_emb_dim: bool,
    ):
        layers = nn.ModuleList([])
        ch = chans
        for _ in range(num_pool_layers - 1):
            layers.append(TimeConvAttentionBlock(ch, ch * 2, time_emb_dim))
            ch *= 2
        return layers

    def create_up_conv_layers(
        self,
        chans: int,
        num_pool_layers: int,
    ):
        layers = nn.ModuleList()
        ch = chans * (2 ** (num_pool_layers - 1))
        for _ in range(num_pool_layers - 1):
            layers.append(nn.ConvTranspose2d(ch, ch, kernel_size=2, stride=2))
            ch //= 2
        layers.append(nn.Identity())
        return layers

    def create_up_sample_layers(
        self,
        chans: int,
        num_pool_layers: int,
        time_emb_dim: bool,
    ):
        layers = nn.ModuleList()
        ch = chans * (2 ** (num_pool_layers - 1))
        for _ in range(num_pool_layers - 1):
            layers.append(
                TimeConvAttentionBlock(
                    ch * 2,
                    ch // 2,
                    time_emb_dim,
                )
            )
            ch //= 2
        layers.append(
            TimeConvAttentionBlock(
                ch * 2,
                ch,
                time_emb_dim,
            )
        )
        return layers

    def forward(
        self,
        x: Tensor,
        t: Tensor,
    ) -> Tensor:
        t_emb = self.time_mlp(t)

        stack = []
        output = self.first_layer(x, t_emb)
        stack.append(output)
        for down_conv, layer in zip(self.down_conv_layers, self.down_sample_layers, strict=False):
            output = layer(output, t_emb)
            stack.append(output)
            output = down_conv(output)

        output = self.bottleneck_conv(output, t_emb)

        for up_conv, layer in zip(self.up_conv_layers, self.up_sample_layers, strict=False):
            downsampled_output = stack.pop()
            output = up_conv(output)

            B, C, W, H = downsampled_output.shape
            output = output[:, :, :W, :H]

            output = torch.cat([output, downsampled_output], dim=1)
            output = layer(output, t_emb)

        output = self.concat_conv(torch.cat([output, x], dim=1), t_emb)
        output = self.final_conv(output, t_emb)
        return output

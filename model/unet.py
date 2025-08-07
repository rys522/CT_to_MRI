import torch
from torch import Tensor

import torch.nn as nn

def create_down_conv_layers(
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
    chans: int,
    num_pool_layers: int,
):
    layers = nn.ModuleList([])
    ch = chans
    for _ in range(num_pool_layers - 1):
        layers.append(TimeConvAttentionBlock(ch, ch * 2, time_emb_dim))
        ch *= 2
    return layers

def create_up_conv_layers(
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
    chans: int,
    num_pool_layers: int,
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

class ResBlock(nn.Module):
    def __init__ (
        self, 
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
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
    
    def forward(self, x):
        identity = self.skip(x)
        gate = self.skip_gate(x)
        identity = identity * gate

        x = self.layer1(x)

        x = self.layer2(x)

        return identity + x
    
class ConvAttentionBlock(nn.Module):
    def __init__(
        self, 
        in_chans: int,
        out_chans: int,
    ):
        super().__init__(*args, **kwargs)
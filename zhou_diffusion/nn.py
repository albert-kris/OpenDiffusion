"""
Various utilities for neural networks.
"""

import math

import torch
import torch.nn as nn

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    
    EMA smooths parameter fluctuations by weighted averaging of historical parameters,
    commonly used during gradient descent.
    
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    Example: loss = mean_flat((x_pred - x_true) ** 2)  # Per-sample mean squared error
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def normalization(channels,num_groups=32):
    """
    Create a standard normalization layer.
    Note: channels must be divisible by num_groups (default 32).
    
    :param channels: number of input channels.
    :param num_groups: number of groups for GroupNorm.
    :return: an nn.Module for normalization.
    """
    # Ensure num_groups divides channels. If not, pick the largest divisor <= num_groups.
    # This avoids ValueError: num_channels must be divisible by num_groups
    ng = min(num_groups, channels)
    while ng > 0:
        if channels % ng == 0:
            return GroupNorm32(num_groups=ng, num_channels=channels)
        ng -= 1
    # Fallback (should never happen since ng == 1 always divides channels)
    return GroupNorm32(num_groups=1, num_channels=channels)

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    Transforms [N,] -> [N,dim]
    
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
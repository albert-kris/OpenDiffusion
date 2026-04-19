from abc import abstractmethod

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.nn import (
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from torch.utils.checkpoint import checkpoint

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    Forces subclasses to accept timestep embedding 'emb' in forward().
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    When multiple parent classes have the same method, Python uses MRO (Method Resolution Order)
    to select the first matching method.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    Uses interpolation for upsampling.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)
        # With stride=3 and padding=1, only changes channels without altering spatial dimensions
    def forward(self, x):  # x: [B, C, H, W]
        assert x.shape[1] == self.channels
        if self.dims == 3:  # For 3D convolution
            x = F.interpolate(  # [N, C, D, H, W] -> [N, C, D, 2H, 2W]
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:  # [N, C, H, W] -> [N, C, 2H, 2W]
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    Uses pooling for downsampling.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:  # Requires H == 2*k for integer k
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        num_groups=32,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        *,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.num_groups = num_groups
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels,self.num_groups),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels,self.num_groups),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, x, emb, use_reentrant=False
        ) if self.use_checkpoint else self._forward(x,emb)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        # emb_out = self.emb_layers(emb).type(h.dtype)
        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim*2, bias=False)
        self.scale = dim ** -0.5

    def forward(self, x, cond_img):
        # x: Generated image features [b, hw, c]
        # cond_img: Conditional image features [b, hw, c]
        b,c,h,w = x.shape
        x = x.reshape(b,c,-1)
        cond_img = cond_img.reshape(b,c,-1)
        q = self.to_q(x)
        k, v = self.to_kv(cond_img).chunk(2, dim=-1)
        
        attn = torch.einsum('bic,bjc->bij', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Weighted aggregation of conditional image information
        out = torch.einsum('bij,bjc->bic', attn, v).reshape(b,c,h,w)
        return out

class AttentionBlock(TimestepBlock):
    """
    An attention block that allows spatial positions to attend to each other.
    Now supports FiLM conditioning via timestep embeddings.
    """

    def __init__(
        self,
        channels,
        emb_channels=None,
        num_heads=1,
        num_head_channels=-1,
        num_groups = 32,
        use_checkpoint=False,
        use_new_attention_order=False,
        use_film=False,
        use_film_on_kv=False,
    ):
        super().__init__()
        self.channels = channels
        self.use_film = use_film
        self.use_film_on_kv = use_film_on_kv
        if num_head_channels == -1:  # If this parameter is not provided
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint  # Reduces training memory via checkpointing
        self.norm = normalization(channels,num_groups)
        
        # FiLM conditioning: generate scale and shift from timestep embedding
        if use_film and emb_channels is not None:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(emb_channels, 2 * channels),
            )
            # Initialize FiLM weights for safe training (near-identity at start)
            nn.init.zeros_(self.emb_layers[1].bias)
            nn.init.normal_(self.emb_layers[1].weight, mean=0.0, std=1e-3)
        
        # FiLM conditioning on K/V: modulate key and value separately
        if use_film_on_kv and emb_channels is not None:
            self.kv_film_layers = nn.Sequential(
                nn.SiLU(),
                linear(emb_channels, 2 * channels * 2),  # 2 channels for K and V, each with scale and shift
            )
            # Initialize FiLM weights for safe training (near-identity at start)
            nn.init.zeros_(self.kv_film_layers[1].bias)
            nn.init.normal_(self.kv_film_layers[1].weight, mean=0.0, std=1e-3)
        
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, emb=None):
        return checkpoint(self._forward, x, emb, use_reentrant=False) if self.use_checkpoint else self._forward(x, emb)

    def _forward(self, x, emb=None):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        
        # Apply normalization
        h_norm = self.norm(x)
        
        # Apply FiLM conditioning if enabled
        if self.use_film and emb is not None:
            emb_out = self.emb_layers(emb)
            while len(emb_out.shape) < len(h_norm.shape):
                emb_out = emb_out[..., None]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h_norm = h_norm * (1 + scale) + shift
        
        qkv = self.qkv(h_norm)
        
        # Apply FiLM conditioning on K/V if enabled
        if self.use_film_on_kv and emb is not None:
            # Split QKV: qkv has shape [B, C*3, T]
            bs, width, length = qkv.shape
            ch = width // 3
            q, k, v = qkv.chunk(3, dim=1)  # Each has shape [B, C, T]
            
            # Generate FiLM parameters for K and V
            kv_film_out = self.kv_film_layers(emb)  # [B, 4*C]
            # Expand to match spatial dimensions
            while len(kv_film_out.shape) < len(k.shape):
                kv_film_out = kv_film_out[..., None]
            
            # Split into scale and shift for K and V
            scale_k, shift_k, scale_v, shift_v = torch.chunk(kv_film_out, 4, dim=1)
            
            # Apply FiLM to K and V
            k = k * (1 + scale_k) + shift_k
            v = v * (1 + scale_v) + shift_v
            
            # Recombine Q, K, V
            qkv = torch.cat([q, k, v], dim=1)
        
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        #weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        weight = torch.softmax(weight, dim=-1)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        #weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        weight = torch.softmax(weight, dim=-1)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

  
class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,  # Image size
        in_channels,  # Input channels
        base_channels,  # Base channel count
        last_base_channels,
        num_groups_last,
        out_channels,  # Output channels
        num_res_blocks,  # Number of residual blocks in up/downsampling
        attention_resolutions,  # At which channel_mult levels to insert attention layers (corresponds to channel_mult parameter)
        dropout=0, 
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,  # Whether to use learnable convolution for up/downsampling
        dims=2, 
        use_checkpoint=False,  # Use checkpoint technique to reduce training memory
        num_heads=1,  # Number of attention heads
        num_groups=32,
        num_head_channels=-1,  # Auto-specified in code, normally don't change, used for attnblock
        num_heads_upsample=-1,  # Auto-specified in code, normally don't change, used for attnblock
        use_scale_shift_norm=False,  # Use FiLM conditioning in residual blocks (scale and shift normalization layer parameters)
        resblock_updown=False,
        time_embed_dim = -1,
        use_new_attention_order=False,
        use_self_conditioning=False,  # Whether to use self-conditioning
        use_film_in_attention=False,  # Whether to use FiLM conditioning in attention blocks
        use_film_on_kv=False,  # Whether to use FiLM conditioning on K/V in attention blocks
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        
        self.use_film_in_attention = use_film_in_attention
        self.use_film_on_kv = use_film_on_kv

        self.image_size = image_size
        self.in_channels = in_channels
        self.use_self_conditioning = use_self_conditioning
        # Double input channels when self-conditioning is enabled
        actual_in_channels = in_channels * 2 if use_self_conditioning else in_channels
        self.base_channels = base_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        # self.dtype = torch.float32
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        
        # Set time_embed_dim before using it
        if time_embed_dim == -1:
            time_embed_dim = base_channels * 4
        self.time_embed_dim = time_embed_dim
        # Set time_embed_dim before using it
        if time_embed_dim == -1:
            time_embed_dim = base_channels * 4
        
        self.img_emb = nn.Sequential(
            conv_nd(dims, in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            conv_nd(dims, 32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            conv_nd(dims, 64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(1),
            nn.Linear(128*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, time_embed_dim)
        )

        self.time_embed = nn.Sequential(
            linear(base_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * base_channels)
        # Use actual_in_channels (doubled when self-conditioning) for first conv
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, actual_in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * base_channels) if level != len(channel_mult) - 1 else int(mult * last_base_channels),
                        dims=dims,
                        num_groups = self.num_groups if level != len(channel_mult) - 1 else num_groups_last,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * base_channels) if level != len(channel_mult) - 1 else int(mult * last_base_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            emb_channels=self.time_embed_dim,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            num_groups = self.num_groups if level != len(channel_mult) - 1 else num_groups_last,
                            use_new_attention_order=use_new_attention_order,
                            use_film=self.use_film_in_attention,
                            use_film_on_kv=self.use_film_on_kv,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:  # Perform downsampling if not the last level
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            num_groups = self.num_groups,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                num_groups = num_groups_last,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                emb_channels=self.time_embed_dim,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                num_groups = num_groups_last,
                use_new_attention_order=use_new_attention_order,
                use_film=self.use_film_in_attention,
                use_film_on_kv=self.use_film_on_kv,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                num_groups = num_groups_last,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            )
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(base_channels * mult),
                        dims=dims,
                        num_groups = self.num_groups if level!=len(channel_mult)-1 else num_groups_last,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(base_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            emb_channels=self.time_embed_dim,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            num_groups = self.num_groups,
                            use_new_attention_order=use_new_attention_order,
                            use_film=self.use_film_in_attention,
                            use_film_on_kv=self.use_film_on_kv,
                        )
                    )
                if level and i == num_res_blocks:  # Last res block in non-final layers
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            num_groups = self.num_groups,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch,self.num_groups),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def forward(self, 
                x, 
                timesteps, 
                return_z=False,
                self_cond=None,
                ):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param ref_img: a image for condition reference.
        :param self_cond: optional self-conditioning input (previous prediction).
        :param return_z: Controls feature return behavior:
                        - False: return only output (default)
                        - True: return (output, z_original) for backward compatibility
                        - 'all': return (output, z_features_dict) with all layer features
        :return: an [N x C x ...] Tensor of outputs, or (output, z) if return_z is True/='all'.
        """
        if self.use_self_conditioning:
            if self_cond is None:
                self_cond = torch.zeros_like(x)
            x = torch.cat([x, self_cond], dim=1)
        
        hs = []
        collect_all_features = (return_z == 'all')
        z_features = {} if collect_all_features else None
        
        emb = self.time_embed(timestep_embedding(timesteps, self.base_channels))
        h = x
        
        for i, module in enumerate(self.input_blocks):
            h = module(h, emb)
            hs.append(h)
            if collect_all_features:
                z_features[f'input_block_{i}'] = h.flatten(1)
        
        h_before_middle = h
        h = self.middle_block(h, emb)
        
        if collect_all_features:
            z_features['middle_before'] = h_before_middle.flatten(1)
            z_features['middle_after'] = h.flatten(1)
        
        z_original = h.flatten(1)
        
        for i, module in enumerate(self.output_blocks):
            h_skip = hs.pop()
            h = torch.cat([h, h_skip], dim=1)
            h = module(h, emb)
            if collect_all_features:
                z_features[f'output_block_{i}'] = h.flatten(1)
        
        h_before_out = h
        out = self.out(h)
        
        if return_z == 'all':
            z_features['final_before'] = h_before_out.flatten(1)
            z_features['z_original'] = z_original
            return out, z_features
        elif return_z:
            return out, z_original
        else:
            return out
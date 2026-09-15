import torch
import torch.nn as nn
from functools import partial

from ldm3d_lapt.modules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    timestep_embedding,
)
from ldm3d_lapt.modules.attention import *
from ldm3d_lapt.modules.encoders import *


class PseudoConv3d(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        kernel_size = 3,
        *,
        temporal_kernel_size = None,
        temporal_length = 32,
        **kwargs
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        temporal_kernel_size = default(temporal_kernel_size, kernel_size)
        self.temporal_length = temporal_length

        self.temporal_conv = nn.Conv1d(dim_out, dim_out, kernel_size = temporal_kernel_size, padding = temporal_kernel_size // 2) if kernel_size > 1 else None

        if exists(self.temporal_conv):
            nn.init.dirac_(self.temporal_conv.weight.data) # initialized to be identity
            nn.init.zeros_(self.temporal_conv.bias.data)

    def forward(
        self,
        x,
        enable_time = True
    ):
        _, _, h, w = x.shape

        if not enable_time or not exists(self.temporal_conv):
            return x
        
        x = rearrange(x, '(b d) c h w -> (b h w) c d', d = self.temporal_length)
        x = self.temporal_conv(x)
        x = rearrange(x, '(b h w) c d -> (b d) c h w', h = h, w = w)

        return x
    

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1, temporal_length=32):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
            self.op_tem = PseudoConv3d(self.out_channels, self.out_channels, 3, temporal_length=temporal_length)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.use_conv:
            x = self.op(x)
            x = self.op_tem(x)
        else:
            x = self.op(x)
        return x


class ResBlock(nn.Module):
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
        dropout,
        out_channels=None,
        use_conv=False,
        dims=2,
        up=False,
        down=False,
        temporal_length=32,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.temporal_length = temporal_length

        self.in_layers = nn.Sequential(
            nn.InstanceNorm2d(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),   
        )
        self.in_layers_tem = PseudoConv3d(channels, self.out_channels, 3, temporal_length=temporal_length)

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.out_layers = nn.Sequential(
            nn.InstanceNorm2d(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )
        self.out_layers_tem = PseudoConv3d(self.out_channels, self.out_channels, 3, temporal_length=temporal_length)

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
            h = self.in_layers_tem(h)
        else:
            h = self.in_layers(x)
            h = self.in_layers_tem(h)
        h = self.out_layers(h)
        h = self.out_layers_tem(h)
        return self.skip_connection(x) + h
    
    
class VolumeConditioner(SpatialRescaler):
    """
    Inherits from SpatialRescaler to first resize the input feature map and then
    encodes spatial position of each slice in a volume, applying a 1D convolution
    along the slice dimension to learn volumetric consistency.
    """
    def __init__(
        self,
        in_channels,
        model_channels,
        dropout=0.0,
        n_stages=1,
        method='bilinear',
        out_channels=None,
        bias=False,
        temporal_length=32
    ):
        
        final_out_channels = model_channels * (2**(n_stages - 1)) if out_channels is None else out_channels
        
        super().__init__(n_stages=n_stages, method=method, multiplier=0.5,
                         in_channels=in_channels, out_channels=final_out_channels, bias=bias)
        
        self.n_stages = n_stages
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)

        self.res_blocks_tem = nn.ModuleList()
        
        current_channels = in_channels

        for i in range(self.n_stages):
            res_block_in_channels = current_channels + in_channels
            resblock_out_channels = model_channels * (2**i)
            
            res_block = ResBlock(
                channels=res_block_in_channels,
                dropout=dropout,
                out_channels=resblock_out_channels,
                use_conv=True,
                dims=2,
                down=True,
                temporal_length=temporal_length,
            )
            self.res_blocks_tem.append(res_block)
            current_channels = resblock_out_channels
        
        self.zero_conv_tem = zero_module(nn.Conv2d(current_channels, final_out_channels, 1))

    def forward(self, x):
        x_rescaled = super().forward(x)

        h = x
        x_orig = x

        for res_block in self.res_blocks_tem:
            if h.shape[-2:] == x_orig.shape[-2:]:
                interp_orig = x_orig
            else:
                interp_orig = self.interpolator(x_orig, size=h.shape[-2:])
            
            inp = torch.cat([h, interp_orig], dim=1)
            h = res_block(inp)
        
        h = self.zero_conv_tem(h)
        x = h + x_rescaled
 
        return x

    def encode(self, x):
        return self(x)
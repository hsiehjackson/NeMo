# add conv, fourier, attn, linear layers

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import torch
from torch import nn as nn
from torch.nn import LayerNorm
import torch.nn.functional as F

from nemo.core.classes.module import NeuralModule
from nemo.collections.asr.parts.submodules.jasper import MaskedConv1d

# LegoBlock is initialized based on list of sub_block configs

__all__ = ['LegoBlock', 'LegoConvSubBlock', 'LegoFourierSubBlock', 'LegoLinearSubBlock']


# Lego block and all the sub-block classes here


class LegoBlock(NeuralModule):
    """A single Lego block. Is initialized using a list config of Lego sub-blocks.

    Args:
        sub_blocks (ListConfig): Sub-blocks.
        d_model (int): input number of channels/model dimension
        (maybe)
        dropout (float): Dropout rate for each sub-block.
        norm (str): Normalization mode for each sub-block.
        outer_residual (bool): Add residual connection from start to end of block.
    """

    def __init__(
            self,
            sub_blocks,
            d_model,  # channels
            dropout=0.,
            outer_residual=False,
    ):
        super(LegoBlock, self).__init__()

        self.sub_blocks = nn.ModuleList()
        self.norms_pre = nn.ModuleList()
        # self.norms_post = nn.ModuleList()

        for sub_cfg in sub_blocks:
            self.norms_pre.append(LayerNorm(d_model))
            # self.norms_pre.append(nn.BatchNorm1d(d_model, eps=1e-3, momentum=0.1))
            # self.norms_post.append(LayerNorm(d_model))
            self.sub_blocks.append(LegoBlock.from_config_dict(sub_cfg))

        self.dropout = nn.Dropout(dropout)

        self.final_norm = LayerNorm(d_model)
        # self.final_norm = nn.BatchNorm1d(d_model, eps=1e-3, momentum=0.1)

        self.outer_residual = outer_residual

        self.activation = nn.ReLU()

    def forward(self, x, lens, att_mask=None, pos_emb=None, pad_mask=None):
        """
        Args:
            x (torch.Tensor): input signals (B, T, d_model)
            att_mask (torch.Tensor): attention masks(B, T, T)
            pos_emb (torch.Tensor): (L, 1, d_model)
            pad_mask (torch.tensor): padding mask
        Returns:
            x (torch.Tensor): (B, T, d_model)
        """

        if self.outer_residual:
            outer_residual = x

        for norm_pre, sub_block in zip(self.norms_pre, self.sub_blocks):
            residual = x
            # x = norm_pre(x.transpose(-2, -1)).transpose(-2, -1)
            x = norm_pre(x)
            if isinstance(sub_block, LegoConvSubBlock):
                x, lens = sub_block(x, lens)
            else:
                x = sub_block(x)
            # x = norm_post(x)
            x = self.activation(x)
            x = self.dropout(x)

            #bad
            if hasattr(sub_block, 'residual_type'):
                if sub_block.residual_type == 'add':
                    x = residual + x
                elif sub_block.residual_type == 'multiply':
                    x = residual * x
            else:
                x = residual + x
                #residual add by default

        x = self.final_norm(x)
        # x = self.final_norm(x.transpose(-2, -1)).transpose(-2, -1)

        if self.outer_residual:
            x = outer_residual + x

        return x, lens


class LegoConvSubBlock(nn.Module):

    def __init__(self, d_model, kernel_size=17, axis="time"):
        super(LegoConvSubBlock, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        self.d_model = d_model

        self.depthwise_conv = MaskedConv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=d_model,
            bias=True,
        )

        self.axis = axis

    def forward(self, x, lens, pad_mask=None):
        if self.axis == "time":
            x = x.transpose(-2, -1)
            if pad_mask is not None:
                x.masked_fill_(pad_mask.unsqueeze(1), 0.0)

        x, lens = self.depthwise_conv(x, lens)

        if self.axis == "time":
            x = x.transpose(-2, -1)

        return x, lens


class LegoFourierSubBlock(nn.Module):

    def __init__(self, dim=-1, norm="ortho", use_fft2=False, patch_size=-1, shift=0,
                 shuffle_after=False, shuffle_groups=8):

        super(LegoFourierSubBlock, self).__init__()

        self.dim = dim
        self.norm = norm
        self.use_fft2 = use_fft2
        self.patch_size = patch_size
        self.shift = shift

        self.shuffle = shuffle_after
        if self.shuffle:
            self.shuffle_op = LegoChannelShuffle(shuffle_groups)

    def forward(self, x, pad_mask=None):
        if x.shape[self.dim] < self.patch_size and self.shift > 0:
            return x

        orig_shape = x.shape
        fft_dim = self.dim
        pad_right = 0

        if self.patch_size != -1:
            # print(x.shape)
            if self.dim != -1:
                x = x.transpose(-1, self.dim)
            orig_shape = x.shape
            fft_dim = -1

            if self.shift > 0:
                x = x[..., self.shift:]

            pad_right = self.patch_size - x.shape[-1] % self.patch_size
            x = F.pad(x, [0, pad_right])

            x = x.reshape(x.shape[:-1] + (x.shape[-1] // self.patch_size, self.patch_size))

            # print(x.shape, fft_dim, self.patch_size, self.dim, self.shift, pad_right)

        x = torch.fft.fft(x, dim=fft_dim, norm=self.norm).real

        if self.shuffle:
            x = self.shuffle_op(x)

        if self.patch_size != -1:
            x = x.reshape(orig_shape[:-1] + (x.shape[-2] * x.shape[-1],))
            x = x[..., :-pad_right]
            x = F.pad(x, [self.shift, 0])
            if self.dim != -1:
                x = x.transpose(-1, self.dim)
            # print(x.shape)

        return x


class LegoLinearSubBlock(nn.Module):

    def __init__(self, d_model, shared_weight_groups=1):
        super(LegoLinearSubBlock, self).__init__()

        assert d_model % shared_weight_groups == 0

        self.group_size = d_model // shared_weight_groups

        self.linear = nn.Linear(self.group_size, self.group_size)

    def forward(self, x):
        original_shape = x.shape
        x = x.view(x.shape[:-1] + (-1, self.group_size))

        x = self.linear(x)

        x = x.view(original_shape)

        return x


class LegoChannelShuffle(nn.Module):

    def __init__(self, groups=8):
        super(LegoChannelShuffle, self).__init__()

        self.groups = groups

    def forward(self, x):
        sh = x.shape

        x = x.reshape(sh[0], sh[1], self.groups, sh[2] // self.groups)
        x = x.transpose(-2, -1)
        x = x.reshape(sh)

        return x


class LegoPartialFourierMod(nn.Module):

    def __init__(self, dim=-1, mod_n=16, complex_linear=False, residual_type='add'):
        super(LegoPartialFourierMod, self).__init__()

        if complex_linear:
            self.lin_r = nn.Linear(mod_n, mod_n)
            #real weights in lin layer
            self.lin_i = nn.Linear(mod_n, mod_n)
            #imaginary weights in lin layer
        else:
            self.lin = nn.Linear(mod_n * 2, mod_n * 2)

        self.complex_linear = complex_linear

        self.mod_n = mod_n
        self.dim = dim

        self.residual_type = residual_type

    def forward(self, x):
        if self.dim != -1:
            x = x.transpose(-1, self.dim)

        h_dim = x.shape[-1]

        f = torch.fft.fft(x)
        f = f[..., :self.mod_n]

        if self.complex_linear:
            new_r = self.lin_r(f.real) - self.lin_i(f.imag)
            new_i = 1j * (self.lin_r(f.imag) + self.lin_i(f.real))
            f_lin = new_r + new_i
            f_lin = F.pad(f_lin, [0, h_dim - self.mod_n])
        else:
            f = torch.view_as_real(f).reshape(*x.shape[:-1], -1)
            f_lin = self.lin(f)
            f_lin = F.pad(f_lin, [0, h_dim * 2 - self.mod_n * 2])
            f_lin = torch.view_as_complex(f_lin.reshape(*f.shape[:-1], -1, 2))

        x_hat = torch.fft.ifft(f_lin).real

        if self.dim != -1:
            x_hat = x_hat.transpose(-1, self.dim)

        return x_hat

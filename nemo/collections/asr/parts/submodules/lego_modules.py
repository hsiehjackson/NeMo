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

import math

# LegoBlock is initialized based on list of sub_block configs

__all__ = ['LegoBlock', 'LegoConvSubBlock', 'LegoFourierSubBlock', 'LegoLinearSubBlock', 'LegoPartialFourierMod']


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
            id=0,
    ):
        super(LegoBlock, self).__init__()

        self.sub_blocks = nn.ModuleList()
        self.norms_pre = nn.ModuleList()
        # self.norms_post = nn.ModuleList()

        for i, sub_block in enumerate(sub_blocks):
            self.norms_pre.append(LayerNorm(d_model))
            # self.norms_pre.append(nn.BatchNorm1d(d_model, eps=1e-3, momentum=0.1))
            # self.norms_post.append(LayerNorm(d_model))
            sub_block.block_id = id
            self.sub_blocks.append(sub_block)

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
            # x = self.activation(x)
            x = self.dropout(x)

            # bad
            if hasattr(sub_block, 'residual_type'):
                if sub_block.residual_type == 'add':
                    x = residual + x
                elif sub_block.residual_type == 'multiply':
                    x = F.sigmoid(x)
                    x = residual * x
            else:
                x = self.activation(x)
                x = residual + x
                # residual add by default

        x = self.final_norm(x)
        # x = self.final_norm(x.transpose(-2, -1)).transpose(-2, -1)

        if self.outer_residual:
            x = outer_residual + x

        return x, lens


class LegoConvSubBlock(nn.Module):

    def __init__(self, d_model, kernel_size=31, axis="time", residual_type="add", stride=1):
        super(LegoConvSubBlock, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        self.d_model = d_model

        self.depthwise_conv = MaskedConv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=d_model,
            bias=True,
        )

        self.axis = axis

        self.residual_type = residual_type

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

        fft_dim = self.dim
        pad_right = 0

        if self.patch_size != -1:
            # #print(x.shape)
            if self.dim != -1:
                x = x.transpose(-1, self.dim)
            orig_shape = x.shape
            fft_dim = -1

            if self.shift > 0:
                x = x[..., self.shift:]

            pad_right = self.patch_size - x.shape[-1] % self.patch_size
            x = F.pad(x, [0, pad_right])

            orig_shape = x.shape

            x = x.reshape(x.shape[:-1] + (x.shape[-1] // self.patch_size, self.patch_size))

        x = torch.fft.rfft(x, dim=fft_dim, norm=self.norm).real

        if self.shuffle:
            x = self.shuffle_op(x)

        if self.patch_size != -1:
            x = x.reshape(orig_shape[:-1] + (x.shape[-2] * x.shape[-1],))
            x = x[..., :-pad_right]
            x = F.pad(x, [self.shift, 0])
            if self.dim != -1:
                x = x.transpose(-1, self.dim)
            # #print(x.shape)

        return x



class LegoLinearSubBlock(nn.Module):

    def __init__(self, d_model, shared_weight_groups=1, f_exp=1, fourier_init=False):
        super(LegoLinearSubBlock, self).__init__()

        assert d_model % shared_weight_groups == 0

        self.group_size = d_model // shared_weight_groups

        if fourier_init == False:
            self.linear = nn.Sequential(nn.Linear(self.group_size, self.group_size * f_exp),
                                    nn.ReLU(),
                                    nn.Linear(self.group_size * f_exp, self.group_size))
        else:
            self.lin_w = nn.Parameter(self.create_fourier_matrix(d_model))
            self.linear_1 = nn.Linear(d_model, d_model)
            self.linear_2 = nn.Linear(d_model, d_model)

        self.fourier_init = fourier_init


    def forward(self, x):
        original_shape = x.shape
        x = x.view(x.shape[:-1] + (-1, self.group_size))

        if self.fourier_init:
            x = self.lin_w @ x
            x = (1 + torch.cos(x.angle())) * x * 0.5
            x = self.linear_1(x.real) + self.linear_2(x.imag)
        else:
            x = self.linear(x)

        x = x.view(original_shape)

        return x

    def create_fourier_matrix(self, n):
        i, j = torch.meshgrid(torch.arange(n), torch.arange(n))
        omega = (-2 * math.pi * 1j / n).exp()
        W = torch.power(omega, i * j) / torch.sqrt(n)
        return W


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


"""def complex_cond_div(x, cond, val):
    x_as_real = torch.view_as_real(x)
    cond = torch.repeat_interleave(cond)
    #0, 1, 2 -> 0, 0, 1, 1, 2, 2

    x_select = x_as_real[cond]
    x_select_complex = torch.view_as_complex(x_select)

    return x_select_complex"""


class LegoPartialFourierMod(nn.Module):

    def __init__(self, dim=-1, mod_n=16, complex_linear=True, residual_type='add', pool=False, f_exp=1,
                 proj_type=1, dim_final=-1, ln_around_freq=False, patch_size=-1, shift=0):
        super(LegoPartialFourierMod, self).__init__()

        if pool:
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            self.pool = None

        self.ln_around_freq = ln_around_freq

        if ln_around_freq:
            self.norm1_r = nn.LayerNorm(mod_n)
            self.norm1_i = nn.LayerNorm(mod_n)
            self.norm2_r = nn.LayerNorm(mod_n)
            self.norm2_i = nn.LayerNorm(mod_n)

        self.lin_r = nn.Linear(mod_n, mod_n * f_exp)
        self.lin_i = nn.Linear(mod_n, mod_n * f_exp)

        self.proj_type = proj_type

        if patch_size != -1:
            dim_final = patch_size

        if proj_type == 1:
            self.lin_r_2 = nn.Linear(mod_n * f_exp, mod_n)
            self.lin_i_2 = nn.Linear(mod_n * f_exp, mod_n)
        else:
            self.lin_r_2 = nn.Linear(mod_n * f_exp, dim_final)
            self.lin_i_2 = nn.Linear(mod_n * f_exp, dim_final)

        self.complex_linear = complex_linear

        self.mod_n = mod_n
        self.dim = dim
        self.patch_size = patch_size

        self.shift = shift

        self.residual_type = residual_type

    def forward(self, x):
        if self.dim != -1:
            x = x.transpose(-1, self.dim)

        if self.pool:
            x = self.pool(x.transpose(-2, -1)).transpose(-2, -1)

        h_dim = x.shape[-1]

        if h_dim < self.mod_n and self.patch_size == -1:
            x = F.pad(x, [0, self.mod_n - h_dim])
            # if using patches will pad regardless

        pad_right = 0

        # print()
        # print(x.shape)

        if self.patch_size != -1:

            shift = self.shift + self.patch_size // 2 * (self.block_id % 2)

            if shift > 0:
                x = x[..., self.shift:]

            pad_right = self.patch_size - x.shape[-1] % self.patch_size
            x = F.pad(x, [0, pad_right])

            orig_shape = x.shape

            # print("or", x.shape)

            x = x.reshape(x.shape[:-1] + (x.shape[-1] // self.patch_size, self.patch_size))
            h_dim = x.shape[-1]

            # print(x.shape)

        f = torch.fft.rfft(x)

        #freq_step = 1 + self.block_id % 4
        #f = f[..., ::freq_step]
        f = f[..., :self.mod_n]

        f_real = f.real
        f_imag = f.imag

        # print(f.shape)

        if self.ln_around_freq:
            f_real = self.norm1_r(f_real)
            f_imag = self.norm1_r(f_imag)

        new_r = self.lin_r(f_real) - self.lin_i(f_imag)
        new_i = 1j * (self.lin_r(f_imag) + self.lin_i(f_real))
        f_lin = new_r + new_i

        f_lin = (1 + torch.cos(f_lin.angle())) * f_lin * 0.5

        if self.proj_type == 1:
            new_r = self.lin_r_2(f_lin.real) - self.lin_i_2(f_lin.imag)
            new_i = self.lin_r_2(f_lin.imag) + self.lin_i_2(f_lin.real)

            if self.ln_around_freq:
                new_r = self.norm2_r(new_r)
                new_i = self.norm2_i(new_i)

            f_lin = new_r + 1j * new_i
            f_lin = F.pad(f_lin, [0, h_dim - self.mod_n])

            x_hat = torch.fft.ifft(f_lin).real

            # print(x_hat.shape)

        else:
            x_hat = self.lin_r_2(f_lin.real) + self.lin_i_2(f_lin.imag)
            if x_hat.shape[-1] < h_dim:
                x_hat = F.pad(x_hat, [0, h_dim - x_hat.shape[-1]])

        x_hat = x_hat[..., :h_dim]

        # print(x_hat.shape)

        if self.patch_size != -1:
            x_hat = x_hat.reshape(orig_shape[:-1] + (x_hat.shape[-2] * x_hat.shape[-1],))
            x_hat = x_hat[..., :-pad_right]
            x_hat = F.pad(x_hat, [self.shift, 0])

            # print(x_hat.shape)

        if self.dim != -1:
            x_hat = x_hat.transpose(-1, self.dim)

        # print(x_hat.shape)
        # print()

        return x_hat

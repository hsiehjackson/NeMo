# coding=utf-8
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import torch
import math
from einops import rearrange
from torch import einsum, nn

__all__ = ['RotaryEmbedding', 'apply_rotary_pos_emb']


# Inverse dim formula to find dim based on number of rotations
def find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings/(num_rotations * 2 * math.pi)))/(2 * math.log(base))

# Find dim range bounds based on rotations
def find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(find_correction_dim(
        low_rot, dim, base, max_position_embeddings))
    high = math.ceil(find_correction_dim(
        high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim-1)  # Clamp values just in case

def linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


class RotaryEmbedding(nn.Module):
    """
    Implements Rotary Position Embedding from https://arxiv.org/abs/2104.09864.
    """

    def __init__(
        self, dim: int, seq_len_interpolation_factor: int = None, pretrained_max_position_embeddings: int = None
    ):
        """
        Args:

            dim (int): rotary embedding dimension
            seq_len_interpolation_factor (int): if not None, discrete positions will be interpolated
            by this factor via the trick in https://arxiv.org/abs/2306.15595.
            pretrained_max_position_embeddings (int): pre-trained max_position_embeddings before position interpolation.
        """
        super().__init__()
        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.pretrained_max_position_embeddings = pretrained_max_position_embeddings

        self.dim=dim
        self.attn_factor=1
        self.extrapolation_factor=1
        self.beta_fast=32
        self.beta_slow=1
        # self.yarn(self.seq_len_interpolation_factor, self.inv_freq.device)
        
    def forward(self, max_seq_len, position_ids=None, offset=0):
        
        # if position_ids is not None:
        #     seq = position_ids.type(torch.float32).to(self.inv_freq.device).squeeze(0)
        # else:
        #     seq = torch.arange(max_seq_len, device=self.inv_freq.device, dtype=torch.float32) + offset

        seq = torch.arange(max_seq_len, device=self.inv_freq.device, dtype=torch.float32) + offset

        # seq *= 1 / self.seq_len_interpolation_factor
        
        # if self.pretrained_max_position_embeddings is not None and self.seq_len_interpolation_factor is not None:
        #     if max_seq_len > self.pretrained_max_position_embeddings * self.seq_len_interpolation_factor:
        #         self.yarn(max_seq_len / (self.pretrained_max_position_embeddings * self.seq_len_interpolation_factor), self.inv_freq.device)
                        
        # if self.pretrained_max_position_embeddings is not None and self.seq_len_interpolation_factor is not None:
        #     if max_seq_len > self.pretrained_max_position_embeddings * self.seq_len_interpolation_factor:
        #         # dynamic ntk aware scaling (length > position we have learned)
        #         # scale = 2 ** math.ceil(math.log(max_seq_len / (self.pretrained_max_position_embeddings * self.seq_len_interpolation_factor), 2) + 1) - 1
        #         scale = 2 * max_seq_len / (self.pretrained_max_position_embeddings * self.seq_len_interpolation_factor) - 1
        #         base = 10000 * (scale ** (self.dim / (self.dim-2)))
        #         inv_freq = 1.0 / ((base ** (torch.arange(0, self.dim, 2, device=self.inv_freq.device).float() / self.dim)))
        #         self.register_buffer("inv_freq", inv_freq)

                # scale = max_seq_len / (self.pretrained_max_position_embeddings * self.seq_len_interpolation_factor)
                # base = 10000 * scale
                # inv_freq = 1.0 / ((base ** (torch.arange(0, self.dim, 2, device=self.inv_freq.device).float() / self.dim)))
                # inv_freq = inv_freq / (scale ** (2 / self.dim))
                # self.register_buffer("inv_freq", inv_freq)
        
        if self.pretrained_max_position_embeddings is not None and self.seq_len_interpolation_factor is not None:
            if max_seq_len > self.pretrained_max_position_embeddings * self.seq_len_interpolation_factor:
                # dynamic linear scaling (length > position we have learned)
                seq *= 1 / (max_seq_len / self.pretrained_max_position_embeddings)
            else:
                # fixed linear scaling
                seq *= 1 / self.seq_len_interpolation_factor

        # freqs = einsum('i , j -> i j', seq, self.inv_freq)
        freqs = torch.outer(seq, self.inv_freq)
        
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        emb = torch.cat((freqs, freqs), dim=-1)
        # emb [seq_length, .., dim]
        return rearrange(emb, 'n d -> n 1 1 d')

    def yarn(self, scale, device):
        
        pos_freqs = 10000 ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scale * pos_freqs)

        low, high = find_correction_range(self.beta_fast, self.beta_slow, self.dim, 10000, self.pretrained_max_position_embeddings)
        inv_freq_mask = (1 - linear_ramp_mask(low, high, self.dim // 2).float().to(device)) * self.extrapolation_factor # Get n-d rotational scaling corrected for extrapolation
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

        self.register_buffer("inv_freq", inv_freq)


def _rotate_half(x):
    """
    change sign so the last dimension
    [A, B, C, D] -> [-C, -D, A, B]
    """
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    """
    input tensor t is of shape [seq_length, ..., dim]
    rotary positional embeding tensor freqs is of shape [seq_length, ..., dim]
    check https://kexue.fm/archives/8265 for detailed formulas
    """
    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    # cos_ = torch.cos(freqs).to(t.dtype)
    # sin_ = torch.sin(freqs).to(t.dtype)
    # t = ((t * cos_) + (_rotate_half(t) * sin_)) 
    t = ((t * freqs.cos()) + (_rotate_half(t) * freqs.sin()))
    return torch.cat((t, t_pass), dim=-1)

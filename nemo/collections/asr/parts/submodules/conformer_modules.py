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

from nemo.collections.asr.parts.submodules.multi_head_attention import (
    MultiHeadAttention,
    RelPositionMultiHeadAttention,
)
from nemo.collections.asr.parts.utils.activations import Swish
from nemo.core.classes.mixins import AccessMixin
from nemo.core.classes.mixins.adapter_mixins import AdapterModuleMixin
from nemo.utils import logging
from nemo.collections.asr.parts.submodules.structured_linear import *

__all__ = ['ConformerConvolution', 'ConformerFeedForward', 'ConformerLayer']


class ConformerLayer(torch.nn.Module, AdapterModuleMixin, AccessMixin):
    """A single block of the Conformer encoder.

    Args:
        d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        d_ff (int): hidden dimension of PositionwiseFeedForward
        n_heads (int): number of heads for multi-head attention
        conv_kernel_size (int): kernel size for depthwise convolution in convolution module
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
    """

    def __init__(
        self,
        d_model,
        d_ff,
        self_attention_model='rel_pos',
        n_heads=4,
        conv_kernel_size=31,
        conv_norm_type='batch_norm',
        dropout=0.1,
        dropout_att=0.1,
        pos_bias_u=None,
        pos_bias_v=None,
        linear_type='standard',
        linear_blocks=4,
        replace_linear_in_attn=False
    ):
        super(ConformerLayer, self).__init__()

        self.self_attention_model = self_attention_model
        self.n_heads = n_heads
        self.fc_factor = 0.5

        # first feed forward module
        self.norm_feed_forward1 = LayerNorm(d_model)
        self.feed_forward1 = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout, linear_type=linear_type,
                                                  linear_blocks=linear_blocks)

        # convolution module
        self.norm_conv = LayerNorm(d_model)
        self.conv = ConformerConvolution(d_model=d_model, kernel_size=conv_kernel_size, norm_type=conv_norm_type)

        # multi-headed self-attention module
        self.norm_self_att = LayerNorm(d_model)
        if replace_linear_in_attn:
            if self_attention_model == 'rel_pos':
                self.self_attn = RelPositionMultiHeadAttention(
                    n_head=n_heads, n_feat=d_model, dropout_rate=dropout_att, pos_bias_u=pos_bias_u, pos_bias_v=pos_bias_v,
                    linear_type=linear_type, linear_blocks=linear_blocks
                )
            elif self_attention_model == 'abs_pos':
                self.self_attn = MultiHeadAttention(n_head=n_heads, n_feat=d_model, dropout_rate=dropout_att,
                                                    linear_type=linear_type, linear_blocks=linear_blocks)
            else:
                raise ValueError(
                    f"'{self_attention_model}' is not not a valid value for 'self_attention_model', "
                    f"valid values can be from ['rel_pos', 'abs_pos']"
                )
        else:
            if self_attention_model == 'rel_pos':
                self.self_attn = RelPositionMultiHeadAttention(
                    n_head=n_heads, n_feat=d_model, dropout_rate=dropout_att, pos_bias_u=pos_bias_u, pos_bias_v=pos_bias_v
                )
            elif self_attention_model == 'abs_pos':
                self.self_attn = MultiHeadAttention(n_head=n_heads, n_feat=d_model, dropout_rate=dropout_att)
            else:
                raise ValueError(
                    f"'{self_attention_model}' is not not a valid value for 'self_attention_model', "
                    f"valid values can be from ['rel_pos', 'abs_pos']"
                )

        # second feed forward module
        self.norm_feed_forward2 = LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout, linear_type=linear_type,
                                                  linear_blocks=linear_blocks)

        self.dropout = nn.Dropout(dropout)
        self.norm_out = LayerNorm(d_model)

    def forward(self, x, att_mask=None, pos_emb=None, pad_mask=None):
        """
        Args:
            x (torch.Tensor): input signals (B, T, d_model)
            att_mask (torch.Tensor): attention masks(B, T, T)
            pos_emb (torch.Tensor): (L, 1, d_model)
            pad_mask (torch.tensor): padding mask
        Returns:
            x (torch.Tensor): (B, T, d_model)
        """
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + self.dropout(x) * self.fc_factor

        x = self.norm_self_att(residual)
        if self.self_attention_model == 'rel_pos':
            x = self.self_attn(query=x, key=x, value=x, mask=att_mask, pos_emb=pos_emb)
        elif self.self_attention_model == 'abs_pos':
            x = self.self_attn(query=x, key=x, value=x, mask=att_mask)
        else:
            x = None
        residual = residual + self.dropout(x)

        x = self.norm_conv(residual)
        x = self.conv(x, pad_mask)
        residual = residual + self.dropout(x)

        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual = residual + self.dropout(x) * self.fc_factor

        x = self.norm_out(residual)

        if self.is_adapter_available():
            # Call the adapters
            x = self.forward_enabled_adapters(x)

        if self.is_access_enabled():
            self.register_accessible_tensor(tensor=x)

        return x


class ConformerConvolution(nn.Module):
    """The convolution module for the Conformer model.
    Args:
        d_model (int): hidden dimension
        kernel_size (int): kernel size for depthwise convolution
    """

    def __init__(self, d_model, kernel_size, norm_type='batch_norm'):
        super(ConformerConvolution, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        self.d_model = d_model

        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_model * 2, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=d_model,
            bias=True,
        )
        if norm_type == 'batch_norm':
            self.batch_norm = nn.BatchNorm1d(d_model)
        elif norm_type == 'layer_norm':
            self.batch_norm = nn.LayerNorm(d_model)
        else:
            raise ValueError(f"conv_norm_type={norm_type} is not valid!")

        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=d_model, out_channels=d_model, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, x, pad_mask=None):
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = nn.functional.glu(x, dim=1)

        if pad_mask is not None:
            x = x.float().masked_fill(pad_mask.unsqueeze(1), 0.0)

        x = self.depthwise_conv(x)

        if isinstance(self.batch_norm, nn.LayerNorm):
            x = x.transpose(1, 2)
            x = self.batch_norm(x)
            x = x.transpose(1, 2)
        else:
            x = self.batch_norm(x)

        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)
        return x


class ConformerFeedForward(nn.Module):
    """
    feed-forward module of Conformer model.
    """

    def __init__(self, d_model, d_ff, dropout, activation=Swish(), linear_type='standard', linear_blocks=4):
        super(ConformerFeedForward, self).__init__()
        self.linear_type = linear_type
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        if linear_type == "dct":
            self.linear = dctLinear(d_model, type=1)
        elif linear_type == "dct2":
            self.linear = nn.Linear(d_model // 2 + 1, d_model)
        elif linear_type == "dct3": #with elemwise mult
            self.linear = nn.Linear(d_model // 2 + 1, d_model)
            self.weight = nn.Parameter(torch.zeros(d_model // 2 + 1, dtype=torch.float))
        elif linear_type == "dct4": #with linear??
            self.linear2 = nn.Linear(d_model // 2 + 1, d_model)
            self.linear1 = nn.Linear(d_model // 2 + 1, d_model // 2 + 1)
        else:
            if linear_type == "standard":
                self.linear1 = nn.Linear(d_model, d_ff)
            elif linear_type == "pixelfly":
                self.linear1 = PixelflyLinear(in_features=d_model, out_features=d_ff,
                                              block_size=d_model // 8, butterfly_size=d_model // 64,
                                              n_factors=2, lowrank_size=d_model // 128)
            else:
                self.linear1 = MonarchLinear(in_features=d_model, out_features=d_ff, nblocks=linear_blocks)

            if linear_type == "standard":
                self.linear2 = nn.Linear(d_ff, d_model)
            elif linear_type == "pixelfly":
                self.linear2 = PixelflyLinear(in_features=d_ff, out_features=d_model,
                                              block_size=d_model // 8, butterfly_size=d_model // 64,
                                              n_factors=2, lowrank_size=d_model // 128)
            else:
                self.linear2 = MonarchLinear(in_features=d_ff, out_features=d_model, nblocks=linear_blocks)

    def forward(self, x):
        if self.linear_type == "dct":
            x = self.linear(x)
            x = self.activation(x)
            x = self.dropout(x)
        elif self.linear_type == "dct2":
            x = torch.fft.rfft(x).real
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear(x)
        elif self.linear_type == "dct3":
            x = x * self.weight
            x = torch.fft.rfft(x).real
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear(x)
        else:
            x = self.linear1(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear2(x)
        return x

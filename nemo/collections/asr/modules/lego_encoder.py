# encoder link conformer but accepts list of Lego layer configs

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

import math
from collections import OrderedDict

import torch
import torch.nn as nn

from nemo.collections.asr.parts.submodules.lego_modules import LegoBlock, LegoConvSubBlock
from nemo.collections.asr.parts.submodules.multi_head_attention import PositionalEncoding, RelPositionalEncoding
from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import AcousticEncodedRepresentation, LengthsType, NeuralType, SpectrogramType
from nemo.collections.asr.parts.submodules.jasper import init_weights
from nemo.collections.asr.parts.submodules.multi_head_attention import (
    MultiHeadAttention,
    RelPositionMultiHeadAttention,
)

from copy import deepcopy

__all__ = ['LegoEncoder']


class LegoEncoder(NeuralModule, Exportable):
    """
    Lego Encoder

    Args:
        sub_blocks (ListConfig): list of configs for the sub-blocks within each block
        feat_in (int): the size of feature channels
        n_blocks (tuple): number of blocks
        d_model (int): the hidden size of the model
        feat_out (int): the size of the output features
            Defaults to -1 (means feat_out is d_model)
        subsampling (str): the method of subsampling, choices=['vggnet', 'striding']
            Defaults to striding.
        subsampling_factor (int): the subsampling factor which should be power of 2
            Defaults to 4.
        subsampling_conv_channels (int): the size of the convolutions in the subsampling module
            Defaults to -1 which would set it to d_model.
        dropout (float): the dropout rate used in all sub-blocks.
            Defaults to 0.1.
        pos_emb_mode (str): type of positional embedding
            'rel_pos': relative positional embedding and Transformer-XL
            'abs_pos': absolute positional embedding and Transformer
            default is rel_pos.
        pos_emb_max_len (int): the maximum length of positional embeddings
            Defaulst to 5000
    """

    def input_example(self):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        input_example = torch.randn(16, self._feat_in, 256).to(next(self.parameters()).device)
        input_example_length = torch.randint(0, 256, (16,)).to(next(self.parameters()).device)
        return tuple([input_example, input_example_length])

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return OrderedDict(
            {
                "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
                "length": NeuralType(tuple('B'), LengthsType()),
            }
        )

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return OrderedDict(
            {
                "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
                "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            }
        )

    def __init__(
            self,
            sub_blocks,
            feat_in,
            n_blocks=(8,),
            d_model=256,
            feat_out=-1,
            subsampling='striding',
            subsampling_factor=4,
            subsampling_conv_channels=-1,
            dropout=0.1,
            pos_emb_mode='abs_pos',
            pos_emb_max_len=5000,
            outer_residual=False,
            multi_block_residual=False,
            multi_block_residual_skip=3,
            conv_stride_every=3,
            conv_stride_total=3,
    ):
        super().__init__()

        self.d_model = d_model
        self._feat_in = feat_in
        self.scale = math.sqrt(self.d_model)

        self.multi_block_residual = multi_block_residual
        self.multi_block_residual_skip = multi_block_residual_skip

        self.conv_stride_every = conv_stride_every
        self.conv_stride_total = conv_stride_total

        self.pre_encode = nn.Linear(feat_in, d_model)

        self.blocks = nn.ModuleList()

        for i in range(len(n_blocks)):
            proto_block = LegoBlock(sub_blocks[i], d_model, outer_residual=outer_residual, dropout=dropout)
            cur_id = 0
            for j in range(n_blocks[i]):
                block = deepcopy(proto_block)  # LegoBlock(sub_blocks, d_model, outer_residual=outer_residual)
                block.id = cur_id
                cur_id += 1
                self.blocks.append(block)

        if feat_out > 0 and feat_out != self.output_dim:
            self.out_proj = nn.Linear(d_model, feat_out)
            self._feat_out = feat_out
        else:
            self.out_proj = None
            self._feat_out = d_model

        self.stride_blocks = nn.ModuleList()
        for i in range(conv_stride_total):
            block = LegoConvSubBlock(d_model, stride=2)
            self.stride_blocks.append(block)

        self.apply(lambda x: init_weights(x, mode='xavier_uniform'))

    @typecheck()
    def forward(self, audio_signal, length=None):
        if length is None:
            length = torch.tensor(audio_signal.size(-1)).repeat(audio_signal.size(0)).to(audio_signal)
        audio_signal = torch.transpose(audio_signal, 1, 2)

        # b, t, d

        if isinstance(self.pre_encode, ConvSubsampling):
            audio_signal, length = self.pre_encode(audio_signal, length)
        else:
            audio_signal = self.pre_encode(audio_signal)

        prev_signal = audio_signal

        strides_done = 0

        for lth, block in enumerate(self.blocks):
            if lth > 0 and self.multi_block_residual and lth % self.multi_block_residual_skip == 0:
                audio_signal += prev_signal
            if lth % self.conv_stride_every == 0 and strides_done < self.conv_stride_total:
                audio_signal, length = self.stride_blocks[strides_done](audio_signal, length)
                strides_done += 1
            audio_signal, length = block(x=audio_signal, lens=length)
            if self.multi_block_residual and lth % self.multi_block_residual_skip == 0:
                prev_signal = audio_signal

        if self.out_proj is not None:
            audio_signal = self.out_proj(audio_signal)

        audio_signal = torch.transpose(audio_signal, 1, 2)
        return audio_signal, length

    @staticmethod
    def make_pad_mask(seq_lens, max_time, device=None):
        """Make masking for padding."""
        bs = seq_lens.size(0)
        seq_range = torch.arange(0, max_time, dtype=torch.int32)
        seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_time)
        seq_lens = seq_lens.type(seq_range_expand.dtype).to(seq_range_expand.device)
        seq_length_expand = seq_lens.unsqueeze(-1)
        mask = seq_range_expand < seq_length_expand

        if device:
            mask = mask.to(device)
        return mask

# ! /usr/bin/python
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

import torch
from torch import nn
import torch.nn.functional as F

from nemo.core import Loss, typecheck
from nemo.core.neural_types import LossType, NeuralType, SpectrogramType, VoidType

import hydra

__all__ = ['SpecReconLoss']



class SpecReconLoss(Loss):
    @property
    def input_types(self):
        """Input types definitions for SpecReconLoss.
        """
        return {
            "spec_in": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "masks": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "out": NeuralType(('B', 'T', 'D'), VoidType()),
        }

    @property
    def output_types(self):
        """Output types definitions for SpecReconLoss.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, in_dim, proj_dim=128, combine_time_steps=1, quantized_targets=True,
                 codebook_size=300, prob_ppl_weight=0.1,
                 num_groups=2, power=1):

        super().__init__()
        self.quantized_targets = quantized_targets
        self.prob_ppl_weight = prob_ppl_weight
        if self.quantized_targets:
            quantizer_cfg = {
                "_target_": "nemo.collections.asr.modules.wav2vec_modules.GumbelVectorQuantizer",
                "dim": in_dim * combine_time_steps,
                "vq_dim": proj_dim,
                "num_vars": codebook_size,
                "groups": num_groups
            }
            self.quantizer = hydra.utils.instantiate(config=quantizer_cfg)
        self.prob_ppl_weight = prob_ppl_weight
        self.combine_time_steps = combine_time_steps

        if not self.quantized_targets:
            self.target_proj = nn.Linear(in_dim * combine_time_steps, proj_dim)

        self.power = power

    @typecheck()
    def forward(self, spec_in, masks, out):
        spec_in = spec_in.transpose(-2, -1)
        masks = masks.transpose(-2, -1)
        targets = spec_in
        # BxTxC

        targets = targets.reshape(targets.shape[0], targets.shape[1] // self.combine_time_steps, -1)
        masks = masks.reshape(targets.shape)
        # targets = self.target_proj(targets)

        if self.quantized_targets:
            targets, prob_ppl_loss, cur_codebook_temp = self.quantizer(targets)
        else:
            targets = self.target_proj(targets)

        masks = torch.round(masks.mean(-1))
        out_masked_only = out[masks]
        targets_masked_only = targets[masks]
        # T'xC
        # number of masked time steps to predict (T')

        sample_size = out_masked_only.shape[0]

        loss = torch.pow(torch.abs(out_masked_only - targets_masked_only), self.power).sum()
        loss /= sample_size

        if self.prob_ppl_weight != 0 and self.quantized_targets:
            prob_ppl_loss = self.prob_ppl_weight * prob_ppl_loss
            loss += prob_ppl_loss

        return loss


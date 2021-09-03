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

from nemo.core import Loss, typecheck
from nemo.core.neural_types import LossType, NeuralType, SpectrogramType, VoidType

__all__ = ['SpecReconLoss']


class SpecReconLoss(Loss):
    @property
    def input_types(self):
        """Input types definitions for SpecRecon.
        """
        return {
            "spec_in": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "masks": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "out": NeuralType(('B', 'T', 'D'), VoidType()),
        }

    @property
    def output_types(self):
        """Output types definitions for SpecRecon.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, only_masked_recon_loss=True, norm_type=None):
        super().__init__()
        self.only_masked_recon_loss = only_masked_recon_loss
        self.norm_type = norm_type

    @typecheck()
    def forward(self, spec_in, masks, out):
        bs = spec_in.shape[0]
        spec_in = spec_in.reshape(bs, -1)
        masks = masks.reshape(bs, -1)
        out = out.reshape(bs, -1)

        if self.only_masked_recon_loss:
            mask_sum = masks.sum(dim=-1)
            spec_diff = torch.linalg.norm((spec_in * masks) - (out * masks), ord=self.norm_type)
            loss = spec_diff / mask_sum
            return loss.mean()
        else:
            spec_diff = torch.linalg.norm((spec_in - out), ord=self.norm_type)
            loss = spec_diff / spec_in.shape[-1]
            return loss.mean()

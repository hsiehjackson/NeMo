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
from nemo.core.neural_types import LossType, NeuralType, SpectrogramType

__all__ = ['MaskedSpecReconLoss']


class MaskedSpecReconLoss(Loss):
    @property
    def input_types(self):
        """Input types definitions for CTCLoss.
        """
        return {
            "spec_in": NeuralType(('B', 'T', 'D'), SpectrogramType()),
            "masks": NeuralType(('B', 'T', 'D'), SpectrogramType()),
            "spec_out": NeuralType(('B', 'T', 'D'), SpectrogramType()),
        }

    @property
    def output_types(self):
        """Output types definitions for CTCLoss.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self):
        super().__init__()

    @typecheck()
    def forward(self, spec_in, masks, spec_out):
        print(spec_in.shape, masks.shape, spec_out.shape)
        mask_sum = masks.sum(dim=(-2, -1))
        spec_diff = torch.abs((spec_in * masks) - (spec_out * masks)).sum(dim=(-2, -1))
        loss = spec_diff / mask_sum
        loss = torch.mean(loss)
        return loss

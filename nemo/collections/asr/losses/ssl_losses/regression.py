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
import torch.nn.functional as F
from torch import nn

from nemo.core import Loss, typecheck
from nemo.core.neural_types import AcousticEncodedRepresentation, LengthsType, LossType, NeuralType, SpectrogramType

__all__ = ["RegressionLoss"]


class RegressionLoss(Loss):
    @property
    def input_types(self):
        """Input types definitions for Contrastive.
        """
        return {
            "spectrograms": NeuralType(("B", "D", "T"), SpectrogramType()),
            "spec_masks": NeuralType(("B", "D", "T"), SpectrogramType()),
            "decoder_outputs": NeuralType(("B", "T", "D"), AcousticEncodedRepresentation()),
            "decoder_lengths": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @property
    def output_types(self):
        """Output types definitions for Contrastive.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    @property
    def needs_labels(self):
        return False

    def __init__(
            self,
            mask_threshold: float = 0.8,
            combine_time_steps: int = 8, #TODO: do this differently
    ):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.mask_threshold = mask_threshold
        self.combine_time_steps = combine_time_steps

    @typecheck()
    def forward(self, spectrograms, spec_masks, decoder_outputs, decoder_lengths=None):
        spectrograms = spectrograms.transpose(1, 2)
        masks = spec_masks.transpose(1, 2)
        # BxTxC

        masks = masks.reshape(masks.shape[0], masks.shape[1] // self.combine_time_steps, -1)
        masks = masks.mean(-1) > self.mask_threshold

        loss = self.mse_loss(spectrograms[masks], decoder_outputs[masks])

        return loss

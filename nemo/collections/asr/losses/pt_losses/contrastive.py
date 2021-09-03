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

import hydra

import torch.nn.functional as F

__all__ = ['ContrastiveLoss']


def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


class ContrastiveLoss(Loss):
    @property
    def input_types(self):
        """Input types definitions for Contrastive.
        """
        return {
            "spec_in": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "masks": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "out": NeuralType(('B', 'T', 'D'), VoidType()),
        }

    @property
    def output_types(self):
        """Output types definitions for Contrastive.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, dim, n_negatives=100, quantized_targets=True, quantizer_cfg=None, prob_ppl_weight=0.1):
        super().__init__()
        self.quantized_targets = quantized_targets
        self.n_negatives = n_negatives
        self.prob_ppl_weight = prob_ppl_weight
        if self.quantized_targets:
            if quantizer_cfg is None:
                quantizer_cfg = {
                    "_target_": "nemo.collections.asr.modules.wav2vec_modules.GumbelVectorQuantizer",
                    "dim": dim,
                    "vq_dim": dim,
                }
            self.quantizer = hydra.utils.instantiate(config=quantizer_cfg)

    def sample_negatives(self, y, num):

        if self.n_negatives == 0:
            return y.new(0)

        bsz, tsz, fsz = y.shape
        y = y.view(-1, fsz)  # BTC => (BxT)C

        high = tsz
        with torch.no_grad():
            assert high > 1, f"{bsz, tsz, fsz}"

            tszs = buffered_arange(num).unsqueeze(-1).expand(-1, self.n_negatives).flatten()

            neg_idxs = torch.randint(low=0, high=high - 1, size=(bsz, self.n_negatives * num))
            neg_idxs[neg_idxs >= tszs] += 1

        for i in range(1, bsz):
            neg_idxs[i] += i * high

        negs = y[neg_idxs.view(-1)]
        negs = negs.view(bsz, num, self.n_negatives, fsz).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs, neg_idxs

    @typecheck()
    def forward(self, spec_in, masks, out):
        spec_in = spec_in.transpose(-2, -1)

        targets = spec_in
        if self.quantized_targets:
            targets, prob_ppl_loss, cur_codebook_temp = self.quantizer(spec_in)
        negatives, _ = self.sample_negatives(targets, targets.size(1))

        # Calculate similarity between logits and all targets, returning FxBxT
        similarity_scores = self._calculate_similarity(out, negatives, targets)

        # Create targets of size B*T
        similarity_targets = out.new_zeros(similarity_scores.size(1) * similarity_scores.size(2), dtype=torch.long)

        # Transpose similarity scores to (T*B)xF for loss
        similarity_scores = similarity_scores.transpose(0, 2)
        similarity_scores = similarity_scores.reshape(-1, similarity_scores.size(-1))

        loss = F.cross_entropy(similarity_scores, similarity_targets, reduction="sum" if self.reduce else "none")

        sample_size = similarity_targets.numel()

        if self.prob_ppl_weight != 0 and self.quantized_targets:
            prob_ppl_loss = self.prob_ppl_weight * prob_ppl_loss * sample_size
            loss += prob_ppl_loss

        return loss

    def _calculate_similarity(self, logits, negatives, targets):
        neg_is_pos = (targets == negatives).all(-1)
        targets = targets.unsqueeze(0)
        targets = torch.cat([targets, negatives], dim=0)
        logits = torch.cosine_similarity(logits.float(), targets.float(), dim=-1).type_as(logits)
        logits /= self.logit_temp
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        return logits

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

    def __init__(self, in_dim, proj_dim=128, combine_time_steps=1, n_negatives=100, quantized_targets=True,
                 codebook_size=300, prob_ppl_weight=0.1,
                 logit_temp=0.1, reduce=True):

        super().__init__()
        self.quantized_targets = quantized_targets
        self.n_negatives = n_negatives
        self.prob_ppl_weight = prob_ppl_weight
        if self.quantized_targets:
            quantizer_cfg = {
                "_target_": "nemo.collections.asr.modules.wav2vec_modules.GumbelVectorQuantizer",
                "dim": in_dim * combine_time_steps,
                "vq_dim": proj_dim,
                "num_vars": codebook_size
            }
            self.quantizer = hydra.utils.instantiate(config=quantizer_cfg)
        self.prob_ppl_weight = prob_ppl_weight
        self.logit_temp = logit_temp
        self.reduce = reduce
        self.combine_time_steps = combine_time_steps

        if not self.quantized_targets:
            self.target_proj = nn.Linear(in_dim * combine_time_steps, proj_dim)

    def sample_negatives(self, y, num):

        if self.n_negatives == 0:
            return y.new(0)

        #bsz, tsz, fsz = y.shape
        #y = y.view(-1, fsz)  # BTC => (BxT)C

        high = y.shape[0]
        with torch.no_grad():
            neg_idxs = torch.randint(low=0, high=high - 1, size=(self.n_negatives * num,))

        negs = y[neg_idxs.view(-1)]
        negs = negs.view(num, self.n_negatives, y.shape[-1]).permute(
            1, 0, 2
        )  # to NxTxC
        return negs, neg_idxs

    @typecheck()
    def forward(self, spec_in, masks, out):
        spec_in = spec_in.transpose(-2, -1)
        masks = masks.transpose(-2, -1)
        targets = spec_in

        targets = targets.reshape(targets.shape[0], targets.shape[1] // self.combine_time_steps, -1)
        masks = masks.reshape(targets.shape)
        #targets = self.target_proj(targets)

        if self.quantized_targets:
            targets, prob_ppl_loss, cur_codebook_temp = self.quantizer(targets)
        else:
            targets = self.target_proj(targets)

        print(masks.shape)
        print(masks.mean(-1)[0])
        masks = masks.mean(-1) > 0.8
        print(masks.shape)
        print(masks[0])
        out_masked_only = out[masks]
        targets_masked_only = targets[masks]
        print(out_masked_only.shape, targets_masked_only.shape)

        negatives, _ = self.sample_negatives(targets.reshape(targets.shape[0] * targets.shape[1], -1),
                                             targets_masked_only.size(0))

        print(negatives.shape)
        # NxTxC

        # Calculate similarity between logits and all targets, returning FxBxT(old)
        similarity_scores = self._calculate_similarity(out_masked_only, negatives, targets_masked_only)
        # FxT ??

        print(similarity_scores.shape)

        # Create targets of size T
        similarity_targets = out.new_zeros(similarity_scores.size(1), dtype=torch.long)
        # T ?

        print(similarity_targets.shape)
        print("-----------")

        # Transpose similarity scores to TxF for loss
        similarity_scores = similarity_scores.transpose(0, 1)

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

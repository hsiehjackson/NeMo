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

import os
import shutil
import tempfile

import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.collections.asr.parts.submodules import rnnt_beam_decoding as beam_decode
from nemo.collections.asr.parts.submodules import rnnt_greedy_decoding as greedy_decode
from nemo.collections.common import tokenizers
from nemo.core.utils import numba_utils
from nemo.core.utils.numba_utils import __NUMBA_MINIMUM_VERSION__
from nemo.utils.app_state import AppState, ModelMetadataRegistry

NUMBA_RNNT_LOSS_AVAILABLE = numba_utils.numba_cpu_is_supported(
    __NUMBA_MINIMUM_VERSION__
) or numba_utils.numba_cuda_is_supported(__NUMBA_MINIMUM_VERSION__)


class TestASRLongAttention:
    @pytest.mark.unit
    def test_forward(self):

        #merge_into_model_config = {
        #    'encoder': dict({'self_attention_model': "longformer_overlap_rel_pos", 'att_context_size': [64, 64]}),
        #}
        #merge_into_model_config = DictConfig(merge_into_model_config)

        # asr_model = ASRModel.from_pretrained('stt_en_conformer_ctc_large', map_location='cuda')

        # asr_model = ASRModel.from_pretrained('stt_en_conformer_transducer_large', map_location='cuda')

        #asr_model = ASRModel.from_pretrained(
        #    'stt_en_conformer_ctc_large', map_location='cuda', merge_into_model_config=merge_into_model_config
        #)

        # asr_model = ASRModel.from_pretrained('stt_en_citrinet_512_gamma_0_25', map_location='cuda')
        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0

        asr_model.compute_eval_loss = False

        device = torch.device('cuda')

        asr_model = asr_model.to(device=device)
        asr_model = asr_model.eval()

        max_len = 16000 * 60 * 29
        input_signal = torch.randn(size=(1, max_len), device=device)
        length = torch.tensor([max_len], device=device)
        print(input_signal.device)

        with torch.no_grad():
            enc = asr_model.forward(input_signal=input_signal, input_signal_length=length)
            # print(enc.shape, enc.device)

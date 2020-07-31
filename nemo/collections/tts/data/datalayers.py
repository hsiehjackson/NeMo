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


import random
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.utils.data
from librosa import istft, stft
from librosa.filters import mel as librosa_mel_fn
from librosa.util import pad_center, tiny
from omegaconf import DictConfig
from scipy.signal import get_window
from torch import nn

from nemo.collections.asr.data.audio_to_text import _AudioDataset
from nemo.collections.asr.parts import features
from nemo.collections.tts.data.text_process import TextProcess
from nemo.collections.tts.data.utils import load_filepaths_and_text, load_wav_to_torch
from nemo.core.neural_types.elements import *
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils.decorators import experimental



@experimental
class AudioToPhonemesDataset(_AudioDataset):
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'audio_signal': NeuralType(
                ('B', 'T'),
                AudioSignal(freq=self._sample_rate)
                if self is not None and hasattr(self, '_sample_rate')
                else AudioSignal(),
            ),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(
        self,
        manifest_filepath: str,
        cmu_dict_path: str,
        featurizer: Union[features.WaveformFeaturizer, features.FilterbankFeatures],
        max_duration: Optional[int] = None,
        min_duration: Optional[int] = None,
        max_utts: int = 0,
        trim: bool = False,
        load_audio: bool = True,
        add_misc: bool = False,
    ):
        self.parser = TextProcess(cmu_dict_path)

        super().__init__(
            manifest_filepath=manifest_filepath,
            featurizer=featurizer,
            parser=self.parser,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            trim=trim,
            load_audio=load_audio,
            add_misc=add_misc,
        )

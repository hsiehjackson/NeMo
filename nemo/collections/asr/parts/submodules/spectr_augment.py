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

import torch
import torch.nn as nn

from nemo.core.classes import Typing, typecheck
from nemo.core.neural_types import LengthsType, NeuralType, SpectrogramType

from nemo.utils import logging


class SpecAugment(nn.Module, Typing):
    """
    Zeroes out(cuts) random continuous horisontal or
    vertical segments of the spectrogram as described in
    SpecAugment (https://arxiv.org/abs/1904.08779).

    params:
    freq_masks - how many frequency segments should be cut
    time_masks - how many time segments should be cut
    freq_width - maximum number of frequencies to be cut in one segment
    time_width - maximum number of time steps to be cut in one segment.
        Can be a positive integer or a float value in the range [0, 1].
        If positive integer value, defines maximum number of time steps
        to be cut in one segment.
        If a float value, defines maximum percentage of timesteps that
        are cut adaptively.
    """

    @property
    def input_types(self):
        """Returns definitions of module input types
        """
        return {
            "input_spec": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output types
        """
        return {"augmented_spec": NeuralType(('B', 'D', 'T'), SpectrogramType())}

    def __init__(
        self,
        freq_masks=0,
        time_masks=0,
        freq_width=10,
        time_width=10,
        rng=None,
        mask_value=0.0,
        same_for_all=False,
        time_min_start=10,
        time_min_width=0,
        freq_min_width=0,
        use_min_len=True,
        snap_time_to_grid=8,
    ):
        super().__init__()

        logging.info("Not using Numba for SpecAugment")

        self._rng = random.Random() if rng is None else rng

        self.freq_masks = freq_masks
        self.time_masks = time_masks

        self.freq_width = freq_width
        self.freq_min_width = freq_min_width
        self.time_width = time_width
        self.time_min_width = time_min_width

        self.mask_value = mask_value

        if isinstance(time_width, int):
            self.adaptive_temporal_width = False
        else:
            if time_width > 1.0 or time_width < 0.0:
                raise ValueError("If `time_width` is a float value, must be in range [0, 1]")

            self.adaptive_temporal_width = True

        self.same_for_all = same_for_all
        self.use_min_len = use_min_len
        self.time_min_start = time_min_start
        self.snap_time_to_grid = snap_time_to_grid

    @typecheck()
    @torch.no_grad()
    def forward(self, input_spec, length):
        sh = input_spec.shape

        if self.same_for_all:

            if self.use_min_len:
                len = min(length)
            else:
                len = sh[2]

            for i in range(self.time_masks):
                if self.adaptive_temporal_width:
                    time_width = max(1, int(len * self.time_width))
                    time_min_width = max(1, int(len * self.time_min_width))
                else:
                    time_width = self.time_width
                    time_min_width = self.time_min_width

                y_left = self._rng.randint(self.time_min_start, max(1, len))
                y_left = y_left + (self.snap_time_to_grid - y_left % self.snap_time_to_grid)
                y_left = min(y_left, len - 1)

                w = self._rng.randint(time_min_width, time_width)
                w = w + (self.snap_time_to_grid - w % self.snap_time_to_grid)

                print(y_left, min(y_left + w, len), w)

                input_spec[:, :, y_left : min(y_left + w, len)] = self.mask_value

            for i in range(self.freq_masks):
                x_left = self._rng.randint(0, sh[1] - self.freq_width)

                w = self._rng.randint(self.freq_min_width, self.freq_width)

                input_spec[:, x_left : x_left + w, :] = self.mask_value

        else:

            for idx in range(sh[0]):
                for i in range(self.freq_masks):
                    x_left = self._rng.randint(0, sh[1] - self.freq_width)

                    w = self._rng.randint(self.freq_min_width, self.freq_width)

                    input_spec[idx, x_left : x_left + w, :] = self.mask_value

                for i in range(self.time_masks):
                    if self.adaptive_temporal_width:
                        time_width = max(1, int(length[idx] * self.time_width))
                        time_min_width = max(1, int(length[idx] * self.time_min_width))
                    else:
                        time_width = self.time_width
                        time_min_width = self.time_min_width

                    y_left = self._rng.randint(self.time_min_start, max(1, length[idx]))
                    y_left = y_left + (self.snap_time_to_grid - y_left % self.snap_time_to_grid)
                    y_left = min(y_left, length[idx] - 1)

                    w = self._rng.randint(time_min_width, time_width)
                    w = w + (self.snap_time_to_grid - w % self.snap_time_to_grid)

                    input_spec[idx, :, y_left : min(y_left + w, length[idx])] = self.mask_value

        return input_spec


class SpecCutout(nn.Module, Typing):
    """
    Zeroes out(cuts) random rectangles in the spectrogram
    as described in (https://arxiv.org/abs/1708.04552).

    params:
    rect_masks - how many rectangular masks should be cut
    rect_freq - maximum size of cut rectangles along the frequency dimension
    rect_time - maximum size of cut rectangles along the time dimension
    """

    @property
    def input_types(self):
        """Returns definitions of module input types
        """
        return {"input_spec": NeuralType(('B', 'D', 'T'), SpectrogramType())}

    @property
    def output_types(self):
        """Returns definitions of module output types
        """
        return {"augmented_spec": NeuralType(('B', 'D', 'T'), SpectrogramType())}

    def __init__(self, rect_masks=0, rect_time=5, rect_freq=20, rng=None):
        super(SpecCutout, self).__init__()

        self._rng = random.Random() if rng is None else rng

        self.rect_masks = rect_masks
        self.rect_time = rect_time
        self.rect_freq = rect_freq

    @typecheck()
    @torch.no_grad()
    def forward(self, input_spec):
        sh = input_spec.shape

        for idx in range(sh[0]):
            for i in range(self.rect_masks):
                rect_x = self._rng.randint(0, sh[1] - self.rect_freq)
                rect_y = self._rng.randint(0, sh[2] - self.rect_time)

                w_x = self._rng.randint(0, self.rect_freq)
                w_y = self._rng.randint(0, self.rect_time)

                input_spec[idx, rect_x : rect_x + w_x, rect_y : rect_y + w_y] = 0.0

        return input_spec

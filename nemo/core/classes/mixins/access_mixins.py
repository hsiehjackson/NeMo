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
from abc import ABC, abstractmethod
from typing import List

from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.asr.parts.utils import asr_module_utils
from nemo.collections.common import tokenizers
from nemo.utils import logging

import torch


_ACCESS_CFG = DictConfig({})

def set_access_cfg(cfg: 'DictConfig'):
    global _ACCESS_CFG
    _ACCESS_CFG = cfg

class AccessMixin(ABC):
    """
    Allows access to output of intermediate layers of a model
    """

    def __init__(self):
        super().__init__()
        self._registry = {}

    def setup_layer_access(self, access_config):
        global _ACCESS_CFG
        _ACCESS_CFG = access_config

    def register_accessible_tensor(
            self, tensor, tensor_name
    ):
        if self.distill_cfg.get('convert_to_cpu', False):
            tensor = tensor.cpu()

        self._registry[tensor] = tensor

    @classmethod
    def get_module_registry(
            cls, module: torch.nn.Module
    ) -> Dict[str, Dict[str, Union[List[Dict[str, torch.Tensor]], List[List[torch.Tensor]]]]]:
        """
        Given a module, will recursively extract in nested lists, all of the registries that may exist.
        The keys of this dictionary are the flattened module names, the values are the internal distillation registry
        of each such module.

        Args:
            module: Any PyTorch Module that extends DistillationMixin.

        Returns:
            A nested dictionary with the following format:
                Dict[Key=module_flattented_name,
                     Value=Dict[Key=loss_name,
                                Value=<list of dictionaries (loss_key: tensor)>  # if keyword loss function
                                      OR
                                      <list of list of tensors>  # if binary loss function
                                ]
                     ]
        """
        module_registry = {}
        for name, m in module.named_modules():
            if hasattr(m, '_registry'):
                module_registry[name] = m._registry
        return module_registry

    @property
    def access_cfg(self):
        """
        Returns:
            The global access config shared across all access mixin modules.
        """
        global _ACCESS_CFG
        return _ACCESS_CFG
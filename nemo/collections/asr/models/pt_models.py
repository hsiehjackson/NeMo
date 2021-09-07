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
import copy
import json
import os
import tempfile
from math import ceil
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from tqdm.auto import tqdm

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.losses.masked_spec_recon_loss import MaskedSpecReconLoss
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.models.asr_model import ASRModel, ExportableEncDecModel
from nemo.collections.asr.parts.mixins import ASRModuleMixin
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, LogprobsType, NeuralType, SpectrogramType
from nemo.utils import logging
from nemo.core.classes import ModelPT

__all__ = ['EncMultiDecPTModel']


class EncMultiDecPTModel(ModelPT, ExportableEncDecModel, ASRModuleMixin):
    """Base class for encoder decoder CTC-based models."""

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        return []

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        # Global_rank and local_rank is set by LightningModule in Lightning 1.2.0
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_gpus

        super().__init__(cfg=cfg, trainer=trainer)
        self.preprocessor = EncMultiDecPTModel.from_config_dict(self._cfg.preprocessor)
        self.encoder = EncMultiDecPTModel.from_config_dict(self._cfg.encoder)

        self.decoders = []
        self.losses = []

        if hasattr(self._cfg, 'loss_alphas'):
            self.loss_alphas = self._cfg.loss_alphas
        else:
            self.loss_alphas = [1.0 for i in range(len(self._cfg.losses))]

        if hasattr(self._cfg, 'loss_log_names'):
            self.loss_log_names = self._cfg.loss_log_names
        else:
            self.loss_log_names = [str(i) for i in range(len(self._cfg.losses))]

        for dec_cfg, loss_cfg in zip(self._cfg.decoders, self._cfg.losses):
            self.decoders.append(EncMultiDecPTModel.from_config_dict(dec_cfg))
            self.losses.append(EncMultiDecPTModel.from_config_dict(loss_cfg))

        # can i use modulelist or do I need multidecoder class?
        self.decoders = nn.ModuleList(self.decoders)
        self.losses = nn.ModuleList(self.losses)

        if hasattr(self._cfg, 'spec_augment') and self._cfg.spec_augment is not None:
            self.spec_augmentation = EncMultiDecPTModel.from_config_dict(self._cfg.spec_augment)
        else:
            self.spec_augmentation = None

        self.masked_evaluation = True

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        # Automatically inject args from model config to dataloader config
        audio_to_text_dataset.inject_dataloader_value_from_model_config(self.cfg, config, key='sample_rate')
        # audio_to_text_dataset.inject_dataloader_value_from_model_config(self.cfg, config, key='labels')
        # config['labels'] = None

        shuffle = config['shuffle']
        device = 'gpu' if torch.cuda.is_available() else 'cpu'

        # Instantiate tarred dataset loader or normal dataset loader
        if config.get('is_tarred', False):
            if ('tarred_audio_filepaths' in config and config['tarred_audio_filepaths'] is None) or (
                    'manifest_filepath' in config and config['manifest_filepath'] is None
            ):
                logging.warning(
                    "Could not load dataset as `manifest_filepath` was None or "
                    f"`tarred_audio_filepaths` is None. Provided config : {config}"
                )
                return None

            shuffle_n = config.get('shuffle_n', 4 * config['batch_size']) if shuffle else 0
            dataset = audio_to_text_dataset.get_tarred_char_dataset(
                config=config,
                shuffle_n=shuffle_n,
                global_rank=self.global_rank,
                world_size=self.world_size,
                augmentor=augmentor,
            )
            shuffle = False
        else:
            if 'manifest_filepath' in config and config['manifest_filepath'] is None:
                logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
                return None

            dataset = audio_to_text_dataset.get_char_dataset(config=config, augmentor=augmentor)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=dataset.collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the training data loader via a Dict-like object.

        Args:
            train_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in train_data_config:
            train_data_config['shuffle'] = True

        # preserve config
        self._update_dataset_config(dataset_name='train', config=train_data_config)

        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if 'is_tarred' in train_data_config and train_data_config['is_tarred']:
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if isinstance(self._trainer.limit_train_batches, float):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches
                    * ceil((len(self._train_dl.dataset) / self.world_size) / train_data_config['batch_size'])
                )

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the validation data loader via a Dict-like object.

        Args:
            val_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='validation', config=val_data_config)

        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the test data loader via a Dict-like object.

        Args:
            test_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='test', config=test_data_config)

        self._test_dl = self._setup_dataloader_from_config(config=test_data_config)

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            input_signal_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            input_signal_eltype = AudioSignal()
        return {
            "input_signal": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
            "input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType(), optional=True),
            "processed_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "outputs": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            "greedy_predictions": NeuralType(('B', 'T'), LabelsType()),
            # "extra": NeuralType(tuple('B'), LengthsType()),
        }

    # @typecheck()
    # how to handle type check for potentially different outputs?
    def forward(
            self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None
    ):
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )

        processed_signal = F.pad(processed_signal, [0, 64 - processed_signal.shape[-1] % 64])

        # processed_signal before spec augment
        spectrograms = processed_signal.detach().clone()
        # spectrograms = F.avg_pool1d(spectrograms, kernel_size=8)

        if self.spec_augmentation is not None and (self.training or self.masked_evaluation):
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        # after spec augment
        masked_spectrograms = processed_signal.detach()
        # masked_spectrograms = F.avg_pool1d(masked_spectrograms, kernel_size=8)
        spec_masks = torch.logical_and(masked_spectrograms < 1e-5, masked_spectrograms > -1e-5).float()
        for idx, proc_len in enumerate(processed_signal_length):
            spec_masks[idx, :, proc_len:] = 0.

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)

        decoder_outs = [decoder(encoder_output=encoded) for decoder in self.decoders]

        return spectrograms, spec_masks, decoder_outs

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        signal, signal_len, transcript, transcript_len = batch
        spectrograms, spec_masks, decoder_outs = self.forward(input_signal=signal, input_signal_length=signal_len)

        return_dict = {'extra': (spectrograms, spec_masks, decoder_outs)}

        log = {'learning_rate': self._optimizer.param_groups[0]['lr']}

        total_loss = 0

        for i, out in enumerate(decoder_outs):
            loss_name = self.loss_log_names[i]
            loss_val = self.losses[i](spec_in=spectrograms,
                                      masks=spec_masks,
                                      out=out)
            log["train_loss_" + loss_name] = loss_val
            total_loss += loss_val * self.loss_alphas[i]

        return_dict['loss'] = total_loss
        log['train_loss'] = total_loss

        return_dict['log'] = log

        return return_dict

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, transcript, transcript_len = batch
        spectrograms, spec_masks, decoder_outs = self.forward(input_signal=signal, input_signal_length=signal_len)

        return_dict = {'extra': (spectrograms, spec_masks, decoder_outs)}

        total_loss = 0

        for i, out in enumerate(decoder_outs):
            loss_name = self.loss_log_names[i]
            loss_val = self.losses[i](spec_in=spectrograms,
                                      masks=spec_masks,
                                      out=out)
            return_dict["val_loss_" + loss_name] = loss_val
            total_loss += loss_val * self.loss_alphas[i]

        return_dict['val_loss'] = total_loss

        return return_dict

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        return_dict = {}
        log = {}

        for key in outputs[0].keys():
            if key[:3] == "val":
                log[key] = torch.stack([x[key] for x in outputs]).mean()

        return_dict['val_loss'] = log['val_loss']
        return_dict['log'] = log

        return return_dict


def test_dataloader(self):
    if self._test_dl is not None:
        return self._test_dl


def _setup_transcribe_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
    """
        Setup function for a temporary data loader which wraps the provided audio file.

        Args:
            config: A python dictionary which contains the following keys:
            paths2audio_files: (a list) of paths to audio files. The files should be relatively short fragments. \
                Recommended length per file is between 5 and 25 seconds.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            temp_dir: (str) A temporary directory where the audio manifest is temporarily
                stored.

        Returns:
            A pytorch DataLoader for the given audio file(s).
        """
    dl_config = {
        'manifest_filepath': os.path.join(config['temp_dir'], 'manifest.json'),
        'sample_rate': self.preprocessor._sample_rate,
        'labels': self.decoder.vocabulary,
        'batch_size': min(config['batch_size'], len(config['paths2audio_files'])),
        'trim_silence': True,
        'shuffle': False,
    }

    temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
    return temporary_datalayer

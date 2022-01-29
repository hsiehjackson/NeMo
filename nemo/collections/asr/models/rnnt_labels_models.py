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
import os
from typing import Dict, List, Optional

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from torch.utils.data import ChainDataset

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.collections.asr.metrics.rnnt_wer_bpe import RNNTBPEWER, RNNTBPEDecoding
from nemo.collections.asr.models.rnnt_models import EncDecRNNTModel
from nemo.collections.asr.parts.mixins import ASRBPEMixin
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging, model_utils


class EncDecRNNTLabelsModel(EncDecRNNTModel):
    """Base class for encoder decoder RNNT-based models for direct training on non-text label sequences."""


    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)


        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)

        super().__init__(cfg=cfg, trainer=trainer)

        # Setup decoding object
        self.decoding = RNNTBPEDecoding(
            decoding_cfg=self.cfg.decoding, decoder=self.decoder, joint=self.joint, tokenizer=self.tokenizer,
        )

        # Setup wer object
        """self.wer = RNNTBPEWER(
            decoding=self.decoding,
            batch_dim_index=0,
            use_cer=self._cfg.get('use_cer', False),
            log_prediction=self._cfg.get('log_prediction', True),
            dist_sync_on_step=True,
        )"""

        # Setup fused Joint step if flag is set
        #if self.joint.fuse_loss_wer:
        #    self.joint.set_loss(self.loss)
        #    self.joint.set_wer(self.wer)

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        shuffle = config['shuffle']

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
            dataset = audio_to_text_dataset.get_tarred_dataset(
                config=config,
                tokenizer=self.tokenizer,
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

            dataset = audio_to_text_dataset.get_bpe_dataset(
                config=config, tokenizer=self.tokenizer, augmentor=augmentor
            )

        if type(dataset) is ChainDataset:
            collate_fn = dataset.datasets[0].collate_fn
        else:
            collate_fn = dataset.collate_fn

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def training_step(self, batch, batch_nb):
        signal, signal_len, transcript, transcript_len = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        # During training, loss must be computed, so decoder forward is necessary
        decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        # Compute full joint and loss
        joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)
        loss_value = self.loss(
            log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
        )

        tensorboard_logs = {'train_loss': loss_value, 'learning_rate': self._optimizer.param_groups[0]['lr']}

        # Log items
        self.log_dict(tensorboard_logs)

        # Preserve batch acoustic model T and language model U parameters if normalizing
        if self._optim_normalize_joint_txu:
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        return {'loss': loss_value}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, transcript, transcript_len = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        tensorboard_logs = {}

        decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)
        joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)

        loss_value = self.loss(
            log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
        )

        tensorboard_logs['val_loss'] = loss_value



        return tensorboard_logs


    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_loss_log = {'val_loss': val_loss_mean}

        tensorboard_logs = {**val_loss_log}

        return {**val_loss_log, 'log': tensorboard_logs}

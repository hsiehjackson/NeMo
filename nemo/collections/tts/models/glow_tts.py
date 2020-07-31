# Copyright 2020 NVIDIA. All Rights Reserved.
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

import math
from typing import Dict, Optional

import torch
import torch.utils.data
from omegaconf import DictConfig
from pytorch_lightning import Trainer

import nemo
from nemo.collections.asr.parts.features import WaveformFeaturizer
from nemo.collections.asr.parts.perturb import process_augmentations
from nemo.collections.tts.data.datalayers import AudioToPhonemesDataset
from nemo.collections.tts.losses.glow_tts_loss import GlowTTSLoss
from nemo.core.classes import ModelPT
from nemo.utils.decorators import experimental


@experimental
class GlowTTSModel(ModelPT):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        super().__init__(cfg=cfg, trainer=trainer)

        self.preprocessor = GlowTTSModel.from_config_dict(self._cfg.preprocessor)

        if hasattr(self._cfg, 'spec_augment') and self._cfg.spec_augment is not None:
            self.spec_augmentation = GlowTTSModel.from_config_dict(self._cfg.spec_augment)
        else:
            self.spec_augmentation = None

        self.encoder = nemo.collections.tts.modules.glow_tts.TextEncoder(
            self.vocab_len,
            cfg.mel_channels,
            cfg.hidden_channels_enc or cfg.hidden_channels,
            cfg.filter_channels,
            cfg.filter_channels_dp,
            cfg.n_heads,
            cfg.n_layers_enc,
            cfg.kernel_size,
            cfg.p_dropout,
            window_size=cfg.window_size,
            mean_only=cfg.mean_only,
            prenet=cfg.prenet,
        )

        self.decoder = nemo.collections.tts.modules.glow_tts.FlowSpecDecoder(
            cfg.mel_channels,
            cfg.hidden_channels_dec or cfg.hidden_channels,
            cfg.kernel_size_dec,
            cfg.dilation_rate,
            cfg.n_blocks_dec,
            cfg.n_block_layers,
            p_dropout=cfg.p_dropout_dec,
            n_sqz=cfg.n_sqz,
            n_split=cfg.n_split,
            sigmoid_scale=cfg.sigmoid_scale,
        )

        self.setup_optimization()

        self.loss = GlowTTSLoss()

    def train_dataloader(self):
        return self._train_dl

    def val_dataloader(self):
        return self._val_dl

    def forward(
        self, x, x_lengths, y=None, y_lengths=None, gen=False, noise_scale=1.0, length_scale=1.0,
    ):

        x_m, x_logs, logw, x_mask = self.encoder(text=x, text_lengths=x_lengths)

        # logw is predicted durations by encoder
        if gen:
            w = torch.exp(logw) * x_mask * length_scale
            w_ceil = torch.ceil(w)
            y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
            y_max_length = None
        else:
            y, y_lengths = self.preprocessor(input_signal=y, length=y_lengths)

            # Spec augment is not applied during evaluation/testing
            if self.spec_augmentation is not None and self.training:
                y = self.spec_augmentation(input_spec=y)

            y_max_length = y.size(2)

            y_max_length = (y_max_length // self._cfg.n_sqz) * self._cfg.n_sqz
            y = y[:, :, :y_max_length]

        y_lengths = (y_lengths // self._cfg.n_sqz) * self._cfg.n_sqz

        y_mask = torch.unsqueeze(nemo.collections.tts.modules.parts.sequence_mask(y_lengths, y_max_length), 1).to(
            x_mask.dtype
        )
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)

        if gen:
            attn = nemo.collections.tts.modules.parts.generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(
                1
            )
            y_m = torch.matmul(attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)).transpose(
                1, 2
            )  # [b, t', t], [b, t, d] -> [b, d, t']
            y_logs = torch.matmul(attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)).transpose(
                1, 2
            )  # [b, t', t], [b, t, d] -> [b, d, t']
            logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask

            z = (y_m + torch.exp(y_logs) * torch.randn_like(y_m) * noise_scale) * y_mask
            y, logdet = self.decoder(spect=z, spect_mask=y_mask, reverse=True)
            return (y, y_m, y_logs, logdet), attn, logw, logw_, x_m, x_logs
        else:
            z, logdet = self.decoder(spect=y, spect_mask=y_mask, reverse=False)

            with torch.no_grad():
                x_s_sq_r = torch.exp(-2 * x_logs)
                logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - x_logs, [1]).unsqueeze(-1)  # [b, t, 1]
                logp2 = torch.matmul(x_s_sq_r.transpose(1, 2), -0.5 * (z ** 2))  # [b, t, d] x [b, d, t'] = [b, t, t']
                logp3 = torch.matmul((x_m * x_s_sq_r).transpose(1, 2), z)  # [b, t, d] x [b, d, t'] = [b, t, t']
                logp4 = torch.sum(-0.5 * (x_m ** 2) * x_s_sq_r, [1]).unsqueeze(-1)  # [b, t, 1]
                logp = logp1 + logp2 + logp3 + logp4  # [b, t, t']

                attn = (
                    nemo.collections.tts.modules.parts.maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()
                )

            y_m = torch.matmul(attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)).transpose(
                1, 2
            )  # [b, t', t], [b, t, d] -> [b, d, t']
            y_logs = torch.matmul(attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)).transpose(
                1, 2
            )  # [b, t', t], [b, t, d] -> [b, d, t']
            logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask
            # logw_ is durations from monotonic alignment

            return z, y_m, y_logs, logdet, logw, logw_, y_lengths

    def training_step(self, batch, batch_idx):

        y, y_lengths, x, x_lengths = batch

        z, y_m, y_logs, logdet, logw, logw_, y_lengths = self(x, x_lengths, y, y_lengths, gen=False)

        l_mle, l_length, logdet = self.loss(
            z=z,
            y_m=y_m,
            y_logs=y_logs,
            logdet=logdet,
            logw=logw,
            logw_=logw_,
            x_lengths=x_lengths,
            y_lengths=y_lengths,
        )

        loss = sum([l_mle, l_length])

        output = {
            "loss": loss,  # required
            "progress_bar": {"l_mle": l_mle, "l_length": l_length, "logdet": logdet},  # optional (MUST ALL BE TENSORS)
            "log": {"loss": loss, "l_mle": l_mle, "l_length": l_length, "logdet": logdet},
        }

        return output

    def validation_step(self, batch, batch_idx):

        output = self.training_step(batch, batch_idx)

        return output

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def _setup_dataloader_from_config(self, cfg: DictConfig):

        if 'augmentor' in cfg:
            augmentor = process_augmentations(cfg['augmentor'])
        else:
            augmentor = None

        featurizer = WaveformFeaturizer(
            sample_rate=cfg['sample_rate'], int_values=cfg.get('int_values', False), augmentor=augmentor
        )
        dataset = AudioToPhonemesDataset(
            manifest_filepath=cfg['manifest_filepath'],
            cmu_dict_path=cfg.get('cmu_dict_path', None),
            featurizer=featurizer,
            max_duration=cfg.get('max_duration', None),
            min_duration=cfg.get('min_duration', None),
            max_utts=cfg.get('max_utts', 0),
            trim=cfg.get('trim_silence', True),
            load_audio=cfg.get('load_audio', True),
            add_misc=cfg.get('add_misc', False),
        )

        self.vocab_len = len(dataset.parser.symbols)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg['batch_size'],
            collate_fn=dataset.collate_fn,
            drop_last=cfg.get('drop_last', False),
            shuffle=cfg['shuffle'],
            num_workers=cfg.get('num_workers', 0),
        )

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        self._val_dl = self._setup_dataloader_from_config(cfg=val_data_config)

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass

    @classmethod
    def from_pretrained(cls, name: str):
        pass

    def export(self, **kwargs):
        pass

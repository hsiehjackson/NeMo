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

from typing import Dict, Optional

import torch
import torch.nn.functional as F
import torch.utils.data
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.asr.parts.perturb import process_augmentations
from nemo.collections.tts.data.datalayers import AudioToPhonemesDataset
from nemo.collections.tts.helpers.helpers import log_audio_to_tb, plot_alignment_to_numpy, \
    plot_spectrogram_to_numpy, plot_pitch_to_numpy
from nemo.collections.tts.losses.glow_tts_loss import GlowTTSLoss
from nemo.collections.tts.modules.glow_tts import GlowTTSModule
from nemo.core.classes import ModelPT
from nemo.utils.decorators import experimental


@experimental
class GlowTTSModel(ModelPT):
    """
    GlowTTS model used to generate spectrograms from text
    Consists of a text encoder and an invertible spectrogram decoder
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        super().__init__(cfg=cfg, trainer=trainer)

        self.preprocessor = GlowTTSModel.from_config_dict(self._cfg.preprocessor)

        encoder = GlowTTSModel.from_config_dict(self._cfg.encoder)
        decoder = GlowTTSModel.from_config_dict(self._cfg.decoder)

        if self._cfg.use_pitch:
            pitch_predictor = GlowTTSModel.from_config_dict(self._cfg.pitch_predictor)
        else:
            pitch_predictor = None

        self.glow_tts = GlowTTSModule(encoder,
                                      decoder,
                                      pitch_predictor=pitch_predictor,
                                      n_speakers=cfg.n_speakers,
                                      speaker_channels=cfg.speaker_channels)

        self.setup_optimization()

        self.loss = GlowTTSLoss()

    def train_dataloader(self):
        return self._train_dl

    def val_dataloader(self):
        return self._val_dl

    def forward(self, x, x_lengths, y=None, y_lengths=None, gen=False, noise_scale=0.3, length_scale=1.0, pitch=None):

        if gen:
            return self.glow_tts.generate_spect(
                text=x, text_lengths=x_lengths, noise_scale=noise_scale, length_scale=length_scale, pitch=pitch
            )
        else:
            return self.glow_tts(text=x, text_lengths=x_lengths, spect=y, spect_lengths=y_lengths, pitch=pitch)

    def step(self, y, y_lengths, x, x_lengths, pitch=None):

        if pitch is not None:
            if pitch.shape[1] < y.shape[2]:
                pitch = F.pad(pitch, [0, y.shape[2] - pitch.shape[1]])
            pitch = pitch[:, :y.shape[2]]

        z, y_m, y_logs, logdet, logw, logw_, y_lengths, attn, pitch_pred = self(x, x_lengths, y, y_lengths, gen=False,
                                                                        pitch=pitch)

        l_mle, l_length, logdet, l_pitch = self.loss(
            z=z,
            y_m=y_m,
            y_logs=y_logs,
            logdet=logdet,
            logw=logw,
            logw_=logw_,
            x_lengths=x_lengths,
            y_lengths=y_lengths,
            pitch_pred=pitch_pred,
            pitch_real=pitch
        )

        loss = sum([l_mle, l_length, l_pitch])

        return l_mle, l_length, logdet, loss, attn, l_pitch

    def training_step(self, batch, batch_idx):

        if self._cfg.use_pitch:
            y, y_lengths, x, x_lengths, pitch = batch
        else:
            y, y_lengths, x, x_lengths = batch
            pitch = None

        y, y_lengths = self.preprocessor(input_signal=y, length=y_lengths)

        l_mle, l_length, logdet, loss, _, l_pitch = self.step(y, y_lengths, x, x_lengths, pitch)

        output = {
            "loss": loss,  # required
            "progress_bar": {"l_mle": l_mle, "l_length": l_length, "logdet": logdet, "l_pitch": l_pitch},
            "log": {"loss": loss, "l_mle": l_mle, "l_length": l_length, "logdet": logdet, "l_pitch": l_pitch},
        }

        return output

    def validation_step(self, batch, batch_idx):

        if self._cfg.use_pitch:
            y, y_lengths, x, x_lengths, pitch = batch
        else:
            y, y_lengths, x, x_lengths = batch
            pitch = None

        y, y_lengths = self.preprocessor(input_signal=y, length=y_lengths)

        l_mle, l_length, logdet, loss, attn, l_pitch = self.step(y, y_lengths, x, x_lengths, pitch)

        y_gen, attn_gen, pitch_pred = self(x, x_lengths, gen=True, pitch=True)

        return {
            "loss": loss,
            "l_mle": l_mle,
            "l_length": l_length,
            "l_pitch": l_pitch,
            "logdet": logdet,
            "y": y,
            "y_gen": y_gen,
            "x": x,
            "attn": attn,
            "attn_gen": attn_gen,
            "pitch": pitch,
            "pitch_pred": pitch_pred,
            "progress_bar": {"l_mle": l_mle, "l_length": l_length, "logdet": logdet, "l_pitch": l_pitch},
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_mle = torch.stack([x['l_mle'] for x in outputs]).mean()
        avg_length_loss = torch.stack([x['l_length'] for x in outputs]).mean()
        avg_pitch_loss = torch.stack([x['l_pitch'] for x in outputs]).mean()
        avg_logdet = torch.stack([x['logdet'] for x in outputs]).mean()
        tensorboard_logs = {
            'val_loss': avg_loss,
            'val_mle': avg_mle,
            'val_length_loss': avg_length_loss,
            'val_logdet': avg_logdet,
            'val_pitch_loss': avg_pitch_loss,
        }
        parser = self.val_dataloader().dataset.parser
        separated_phonemes = "|".join([parser.symbols[c] for c in outputs[0]['x'][0]])
        self.logger.experiment.add_text("separated phonemes", separated_phonemes, self.global_step)
        self.logger.experiment.add_image(
            "real_spectrogram",
            plot_spectrogram_to_numpy(outputs[0]['y'][0].data.cpu().numpy()),
            self.global_step,
            dataformats="HWC",
        )
        self.logger.experiment.add_image(
            "generated_spectrogram",
            plot_spectrogram_to_numpy(outputs[0]['y_gen'][0].data.cpu().numpy()),
            self.global_step,
            dataformats="HWC",
        )
        self.logger.experiment.add_image(
            "alignment_for_real_sp",
            plot_alignment_to_numpy(outputs[0]['attn'][0].data.cpu().numpy()),
            self.global_step,
            dataformats="HWC",
        )
        self.logger.experiment.add_image(
            "alignment_for_generated_sp",
            plot_alignment_to_numpy(outputs[0]['attn_gen'][0].data.cpu().numpy()),
            self.global_step,
            dataformats="HWC",
        )
        if self._cfg.use_pitch:
            self.logger.experiment.add_image(
                "real_pitch",
                plot_pitch_to_numpy(outputs[0]['pitch'][0].data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            self.logger.experiment.add_image(
                "predicted_pitch",
                plot_pitch_to_numpy(outputs[0]['pitch_pred'][0].data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
        log_audio_to_tb(self.logger.experiment, outputs[0]['y'][0], "true_audio_gf", self.global_step)
        log_audio_to_tb(self.logger.experiment, outputs[0]['y_gen'][0], "generated_audio_gf", self.global_step)
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def _setup_dataloader_from_config(self, cfg: DictConfig):

        if 'augmentor' in cfg:
            augmentor = process_augmentations(cfg['augmentor'])
        else:
            augmentor = None

        dataset = AudioToPhonemesDataset(
            manifest_filepath=cfg['manifest_filepath'],
            cmu_dict_path=cfg.get('cmu_dict_path', None),
            sample_rate=cfg['sample_rate'],
            int_values=cfg.get('int_values', False),
            augmentor=augmentor,
            max_duration=cfg.get('max_duration', None),
            min_duration=cfg.get('min_duration', None),
            max_utts=cfg.get('max_utts', 0),
            trim=cfg.get('trim_silence', True),
            load_audio=cfg.get('load_audio', True),
            add_misc=cfg.get('add_misc', False),
        )

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

    def export(self, **kwargs):
        pass

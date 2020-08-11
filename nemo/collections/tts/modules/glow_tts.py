# MIT License
#
# Copyright (c) 2020 Jaehyeon Kim
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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

import torch
import torch.nn.functional as F
from torch import nn

from nemo.collections.tts.modules import glow_tts_submodules
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types.elements import *
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils.decorators import experimental


@experimental
class TextEncoder(NeuralModule):
    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        filter_channels_dp: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        window_size: int,
        mean_only: bool = False,
        prenet: bool = False,
        speaker_channels: int = 0,
    ):
        """
        GlowTTS text encoder. Takes in the input text tokens and produces prior distribution statistics for the latent
        representation corresponding to each token, as well as the log durations (the Duration Predictor is also part of
         this module).
        Architecture is similar to Transformer TTS with slight modifications.
        Args:
            n_vocab (int): Number of tokens in the vocabulary
            out_channels (int): Latent representation channels
            hidden_channels (int): Number of channels in the intermediate representations
            filter_channels (int): Number of channels for the representations in the feed-forward layer
            filter_channels_dp (int): Number of channels for the representations in the duration predictor
            n_heads (int): Number of attention heads
            n_layers (int): Number of transformer layers
            kernel_size (int): Kernels size for the feed-forward layer
            p_dropout (float): Dropout probability
            mean_only (bool): Return zeros for logs if true
            prenet (bool): Use an additional network before the transformer modules
            speaker_channels (int): Number of channels in speaker embeddings
        """
        super().__init__()

        self.n_layers = n_layers
        self.hidden_channels = hidden_channels
        self.prenet = prenet
        self.mean_only = mean_only

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        if prenet:
            self.pre = glow_tts_submodules.ConvReluNorm(
                hidden_channels, hidden_channels, hidden_channels, kernel_size=5, n_layers=3, p_dropout=0.5,
            )

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        for i in range(self.n_layers):
            self.attn_layers.append(
                glow_tts_submodules.AttentionBlock(
                    hidden_channels, hidden_channels, n_heads, window_size=window_size, p_dropout=p_dropout,
                )
            )
            self.norm_layers_1.append(glow_tts_submodules.LayerNorm(hidden_channels))
            self.ffn_layers.append(
                glow_tts_submodules.FeedForwardNetwork(
                    hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout
                )
            )
            self.norm_layers_2.append(glow_tts_submodules.LayerNorm(hidden_channels))

        self.proj_m = nn.Conv1d(hidden_channels, out_channels, 1)
        if not mean_only:
            self.proj_s = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj_w = glow_tts_submodules.DurationPredictor(
            hidden_channels + speaker_channels, filter_channels_dp, kernel_size, p_dropout
        )

    @property
    def input_types(self):
        return {
            "text": NeuralType(('B', 'T'), TokenIndex()),
            "text_lengths": NeuralType(('B',), LengthsType()),
            "speaker_embeddings": NeuralType(('B', 'D'), AcousticEncodedRepresentation(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "x_m": NeuralType(('B', 'D', 'T'), NormalDistributionMeanType()),
            "x_logs": NeuralType(('B', 'D', 'T'), NormalDistributionLogVarianceType()),
            "logw": NeuralType(('B', 'T'), TokenLogDurationType()),
            "x_mask": NeuralType(('B', 'D', 'T'), MaskType()),
        }

    @typecheck()
    def forward(self, text, text_lengths, speaker_embeddings=None):

        x = self.emb(text) * math.sqrt(self.hidden_channels)  # [b, t, h]

        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(glow_tts_submodules.sequence_mask(text_lengths, x.size(2)), 1).to(x.dtype)

        if self.prenet:
            x = self.pre(x, x_mask)

        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        for i in range(self.n_layers):
            x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask

        if speaker_embeddings is not None:
            g_exp = speaker_embeddings.expand(-1, -1, x.size(-1))
            x_dp = torch.cat([torch.detach(x), g_exp], 1)
        else:
            x_dp = torch.detach(x)

        x_m = self.proj_m(x) * x_mask
        if not self.mean_only:
            x_logs = self.proj_s(x) * x_mask
        else:
            x_logs = torch.zeros_like(x_m)

        logw = self.proj_w(x=x_dp, mask=x_mask)

        return x_m, x_logs, logw, x_mask

    def save_to(self, save_path: str):
        """TODO: Implement"""
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        """TODO: Implement"""
        pass


@experimental
class FlowSpecDecoder(NeuralModule):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_blocks: int,
        n_layers: int,
        p_dropout: float = 0.0,
        n_split: int = 4,
        n_sqz: int = 2,
        sigmoid_scale: bool = False,
        speaker_channels: int = 0,
        pitch_channels: int = 0,
    ):
        """
        Flow-based invertible decoder for GlowTTS. Converts spectrograms to latent representations and back.
        Args:
            in_channels (int): Number of channels in the input spectrogram
            hidden_channels (int): Number of channels in the intermediate representations
            kernel_size (int): Kernel size in the coupling blocks
            dilation_rate (int): Dilation rate in the WaveNet-like blocks
            n_blocks (int): Number of flow blocks
            n_layers (int): Number of layers within each coupling block
            p_dropout (float): Dropout probability
            n_split (int): Group size for the invertible convolution
            n_sqz (int): The rate by which the spectrograms are squeezed before applying the flows
            sigmoid_scale (bool): Apply sigmoid to logs in the coupling blocks
            speaker_channels (int): Number of channels for speaker embedding
            pitch_channels (int): Number of channels for pitch embedding
        """
        super().__init__()

        self.n_sqz = n_sqz

        if pitch_channels > 0:
            self.pitch_emb = glow_tts_submodules.ConvReluNorm(
                1, hidden_channels, pitch_channels, kernel_size=5, n_layers=3, p_dropout=0.1,
            )

        self.flows = nn.ModuleList()
        for b in range(n_blocks):
            self.flows.append(glow_tts_submodules.ActNorm(channels=in_channels * n_sqz))
            self.flows.append(glow_tts_submodules.InvConvNear(channels=in_channels * n_sqz, n_split=n_split))
            self.flows.append(
                glow_tts_submodules.CouplingBlock(
                    in_channels * n_sqz,
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate,
                    n_layers=n_layers,
                    p_dropout=p_dropout,
                    sigmoid_scale=sigmoid_scale,
                    gin_channels=speaker_channels + pitch_channels,
                )
            )

    @property
    def input_types(self):
        return {
            "spect": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "spect_mask": NeuralType(('B', 'D', 'T'), MaskType()),
            "speaker_embeddings": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation(), optional=True),
            "pitch": NeuralType(('B', 'T'), PitchType(), optional=True),
            "reverse": NeuralType(elements_type=IntType(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "z": NeuralType(('B', 'D', 'T'), NormalDistributionSamplesType()),
            "logdet_tot": NeuralType(('B',), LogDeterminantType()),
        }

    @typecheck()
    def forward(self, spect, spect_mask, speaker_embeddings=None, pitch=None, reverse=False):
        if not reverse:
            flows = self.flows
            logdet_tot = 0
        else:
            flows = reversed(self.flows)
            logdet_tot = None

        x = spect
        x_mask = spect_mask

        cond_emb = None

        if speaker_embeddings is not None:
            cond_emb = speaker_embeddings

        if pitch is not None:
            pitch = pitch.unsqueeze(1)
            p_emb = self.pitch_emb(pitch, x_mask)
            if cond_emb is None:
                cond_emb = p_emb
            else:
                cond_emb = torch.cat((cond_emb, p_emb), dim=1)


        if self.n_sqz > 1:
            x_mask_org = x_mask
            x, x_mask = self.squeeze(x, x_mask_org, self.n_sqz)
            cond_emb, _ = self.squeeze(cond_emb, x_mask_org, self.n_sqz)

        for f in flows:
            if not reverse:
                x, logdet = f(x, x_mask, g=cond_emb, reverse=reverse)
                logdet_tot += logdet
            else:
                x, logdet = f(x, x_mask, g=cond_emb, reverse=reverse)

        if self.n_sqz > 1:
            x, x_mask = self.unsqueeze(x, x_mask, self.n_sqz)

        return x, logdet_tot

    def squeeze(self, x, x_mask=None, n_sqz=2):
        b, c, t = x.size()

        t = (t // n_sqz) * n_sqz
        x = x[:, :, :t]
        x_sqz = x.view(b, c, t // n_sqz, n_sqz)
        x_sqz = x_sqz.permute(0, 3, 1, 2).contiguous().view(b, c * n_sqz, t // n_sqz)

        if x_mask is not None:
            x_mask = x_mask[:, :, n_sqz - 1 :: n_sqz]
        else:
            x_mask = torch.ones(b, 1, t // n_sqz).to(device=x.device, dtype=x.dtype)
        return x_sqz * x_mask, x_mask

    def unsqueeze(self, x, x_mask=None, n_sqz=2):
        b, c, t = x.size()

        x_unsqz = x.view(b, n_sqz, c // n_sqz, t)
        x_unsqz = x_unsqz.permute(0, 2, 3, 1).contiguous().view(b, c // n_sqz, t * n_sqz)

        if x_mask is not None:
            x_mask = x_mask.unsqueeze(-1).repeat(1, 1, 1, n_sqz).view(b, 1, t * n_sqz)
        else:
            x_mask = torch.ones(b, 1, t * n_sqz).to(device=x.device, dtype=x.dtype)
        return x_unsqz * x_mask, x_mask

    def store_inverse(self):
        for f in self.flows:
            f.store_inverse()

    def save_to(self, save_path: str):
        """TODO: Implement"""
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        """TODO: Implement"""
        pass

@experimental
class PitchPredictor(NeuralModule):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            filter_channels: int,
            n_heads: int,
            n_layers: int,
            kernel_size: int,
            p_dropout: float,
            window_size: int,
    ):
        #TODO docstring

        super().__init__()

        self.pre = glow_tts_submodules.ConvReluNorm(
                in_channels, hidden_channels, hidden_channels, kernel_size=5, n_layers=3, p_dropout=0.1,
            )

        self.n_layers = n_layers

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        for i in range(self.n_layers):
            self.attn_layers.append(
                glow_tts_submodules.AttentionBlock(
                    hidden_channels, hidden_channels, n_heads, window_size=window_size, p_dropout=p_dropout,
                )
            )
            self.norm_layers_1.append(glow_tts_submodules.LayerNorm(hidden_channels))
            self.ffn_layers.append(
                glow_tts_submodules.FeedForwardNetwork(
                    hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout
                )
            )
            self.norm_layers_2.append(glow_tts_submodules.LayerNorm(hidden_channels))

        self.proj_final = nn.Conv1d(hidden_channels, 1, 1)

    @typecheck()
    def forward(self, y, y_mask):


        y = self.pre(y, y_mask)


        attn_mask = y_mask.unsqueeze(2) * y_mask.unsqueeze(-1)
        for i in range(self.n_layers):
            y = y * y_mask
            y = self.attn_layers[i](y, y, attn_mask)
            y = self.drop(y)
            y = self.norm_layers_1[i](y + y)

            y = self.ffn_layers[i](y, y_mask)
            y = self.drop(y)
            y = self.norm_layers_2[i](y + y)

        y = y * y_mask

        y = self.proj_final(y) * y_mask

        pitch = F.relu(y.squeeze(1)) * 100.

        return pitch

    @property
    def input_types(self):
        return {
            "y": NeuralType(('B', 'D', 'T'), NormalDistributionMeanType()),
            "y_mask": NeuralType(('B', 'D', 'T'), MaskType()),
        }

    @property
    def output_types(self):
        return {
            "pitch": NeuralType(('B', 'T'), PitchType()),
        }

    def save_to(self, save_path: str):
        """TODO: Implement"""
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        """TODO: Implement"""
        pass


class GlowTTSModule(NeuralModule):
    def __init__(
        self, encoder_module: NeuralModule, decoder_module: NeuralModule, n_speakers: int = 1, speaker_channels: int = 0,
            pitch_predictor: NeuralModule = None
    ):
        """
        Main GlowTTS module. Contains the encoder and decoder.
        Args:
            encoder_module (NeuralModule): Text encoder for predicting latent distribution statistics
            decoder_module (NeuralModule): Invertible spectrogram decoder
            n_speakers (int): Number of speakers
            speaker_channels (int): Channels in speaker embeddings
            pitch_predictor (NeuralModule) Module for predicting pitch if pitch conditioning is turned on
        """
        super().__init__()

        self.encoder = encoder_module
        self.decoder = decoder_module

        self.pitch_predictor = pitch_predictor

        if n_speakers > 1:
            self.speaker_embed = nn.Embedding(n_speakers, speaker_channels)
            nn.init.uniform_(self.emb_g.weight, -0.1, 0.1)


    @property
    def input_types(self):
        return {
            "text": NeuralType(('B', 'T'), TokenIndex()),
            "text_lengths": NeuralType(('B',), LengthsType()),
            "spect": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "spect_lengths": NeuralType(('B',), LengthsType()),
            "speaker": NeuralType(('B',), IntType(), optional=True),
            "pitch": NeuralType(('B', 'T'), PitchType(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "z": NeuralType(('B', 'D', 'T'), NormalDistributionSamplesType()),
            "y_m": NeuralType(('B', 'D', 'T'), NormalDistributionMeanType()),
            "y_logs": NeuralType(('B', 'D', 'T'), NormalDistributionLogVarianceType()),
            "logdet": NeuralType(('B',), LogDeterminantType()),
            "log_durs_predicted": NeuralType(('B', 'T'), TokenLogDurationType()),
            "log_durs_extracted": NeuralType(('B', 'T'), TokenLogDurationType()),
            "spect_lengths": NeuralType(('B',), LengthsType()),
            "attn": NeuralType(('B', 'T', 'T'), SequenceToSequenceAlignmentType()),
            "pitch_pred": NeuralType(('B', 'T'), PitchType(), optional=True),
        }

    @typecheck()
    def forward(self, text, text_lengths, spect, spect_lengths, speaker=None, pitch=None):


        s_emb = None
        if speaker is not None:
            s_emb = F.normalize(self.speaker_embed(speaker)).unsqueeze(-1)

        x_m, x_logs, log_durs_predicted, x_mask = self.encoder(
            text=text, text_lengths=text_lengths, speaker_embeddings=s_emb
        )

        y_max_length = spect.size(2)
        y_max_length = (y_max_length // self.decoder.n_sqz) * self.decoder.n_sqz
        spect = spect[:, :, :y_max_length]

        spect_lengths = (spect_lengths // self.decoder.n_sqz) * self.decoder.n_sqz

        y_mask = torch.unsqueeze(glow_tts_submodules.sequence_mask(spect_lengths, y_max_length), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)

        z, logdet = self.decoder(spect=spect, spect_mask=y_mask, speaker_embeddings=s_emb, pitch=pitch, reverse=False)

        with torch.no_grad():
            x_s_sq_r = torch.exp(-2 * x_logs)
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - x_logs, [1]).unsqueeze(-1)  # [b, t, 1]
            logp2 = torch.matmul(x_s_sq_r.transpose(1, 2), -0.5 * (z ** 2))  # [b, t, d] x [b, d, t'] = [b, t, t']
            logp3 = torch.matmul((x_m * x_s_sq_r).transpose(1, 2), z)  # [b, t, d] x [b, d, t'] = [b, t, t']
            logp4 = torch.sum(-0.5 * (x_m ** 2) * x_s_sq_r, [1]).unsqueeze(-1)  # [b, t, 1]
            logp = logp1 + logp2 + logp3 + logp4  # [b, t, t']

            attn = (glow_tts_submodules.maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()).squeeze(1)

        y_m = torch.matmul(x_m, attn)
        y_logs = torch.matmul(x_logs, attn)

        
        if pitch is not None:
            pitch_pred = self.pitch_predictor(y=y_m, y_mask=y_mask)
        else:
            pitch_pred = None

        log_durs_extracted = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask.squeeze()


        return z, y_m, y_logs, logdet, log_durs_predicted, log_durs_extracted, spect_lengths, attn, pitch_pred

    def generate_spect(self, text, text_lengths, speaker=None, noise_scale=0.3, length_scale=1.0, pitch=False):

        if speaker is not None:
            s_emb = F.normalize(self.speaker_embed(speaker)).unsqueeze(-1)
        else:
            s_emb = None

        x_m, x_logs, log_durs_predicted, x_mask = self.encoder(
            text=text, text_lengths=text_lengths, speaker_embeddings=s_emb
        )

        w = torch.exp(log_durs_predicted) * x_mask.squeeze() * length_scale
        w_ceil = torch.ceil(w)
        spect_lengths = torch.clamp_min(torch.sum(w_ceil, [1]), 1).long()
        y_max_length = None

        spect_lengths = (spect_lengths // self.decoder.n_sqz) * self.decoder.n_sqz

        y_mask = torch.unsqueeze(glow_tts_submodules.sequence_mask(spect_lengths, y_max_length), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)

        attn = glow_tts_submodules.generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1))

        y_m = torch.matmul(x_m, attn)
        y_logs = torch.matmul(x_logs, attn)

        if pitch:
            pitch_pred = self.pitch_predictor(y=y_m, y_mask=y_mask)
        else:
            pitch_pred = None

        z = (y_m + torch.exp(y_logs) * torch.randn_like(y_m) * noise_scale) * y_mask
        y, _ = self.decoder(spect=z, spect_mask=y_mask, speaker_embeddings=s_emb, reverse=True, pitch=pitch_pred)

        return y, attn, pitch_pred

    def save_to(self, save_path: str):
        """TODO: Implement"""
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        """TODO: Implement"""
        pass

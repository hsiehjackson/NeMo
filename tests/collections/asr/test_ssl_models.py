import copy

import pytest
import torch
from omegaconf import DictConfig, ListConfig

from nemo.collections.asr.models import SpeechEncDecSelfSupervisedModel


@pytest.fixture()
def ssl_models():
    preprocessor = {'cls': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor', 'params': dict({})}

    model_defaults = {'enc_hidden': 1024, 'pred_hidden': 64, 'dec_out': 128}

    encoder = {
        'cls': 'nemo.collections.asr.modules.ConvASREncoder',
        'params': {
            'feat_in': 64,
            'activation': 'relu',
            'conv_mask': True,
            'jasper': [
                {
                    'filters': model_defaults['enc_hidden'],
                    'repeat': 1,
                    'kernel': [1],
                    'stride': [1],
                    'dilation': [1],
                    'dropout': 0.0,
                    'residual': False,
                    'separable': True,
                    'se': True,
                    'se_context_size': -1,
                }
            ],
        },
    }

    spec_augment = {
        '_target_': 'nemo.collections.asr.modules.MaskedPatchAugmentation',
        'freq_masks': 3,
        'freq_width': 20,
        'patch_size': 16,
        'mask_patches': 0.5,
    }

    loss_list_contr_mlm = {
        'contr': {
            'decoder': {
                '_target_': 'nemo.collections.asr.modules.ConvASRDecoderReconstruction',
                'feat_in': model_defaults['enc_hidden'],
                'feat_hidden': 128,
                'feat_out': model_defaults['dec_out'],
                'stride_layers': 2,
                'non_stride_layers': 0,
                'stride_transpose': False,
            },
            'loss': {
                '_target_': 'nemo.collections.asr.losses.ContrastiveLoss',
                'in_dim': 80,
                'proj_dim': model_defaults['dec_out'],
                'combine_time_steps': 4,
                'quantized_targets': True,
                'codebook_size': 300,
                'sample_from_same_utterance_only': True,
                'sample_from_non_masked': False,
            },
        },
        'mlm': {
            'decoder': {
                '_target_': 'nemo.collections.asr.modules.ConvASRDecoder',
                'feat_in': model_defaults['enc_hidden'],
                'num_classes': 90000,
            },
            'loss': {'_target_': 'nemo.collections.asr.losses.MLMLoss', 'combine_time_steps': 4},
        },
    }

    modelConfig_contr_mlm = DictConfig(
        {
            'preprocessor': DictConfig(preprocessor),
            'spec_augment': DictConfig(spec_augment),
            'model_defaults': DictConfig(model_defaults),
            'encoder': DictConfig(encoder),
            'loss_list': DictConfig(loss_list_contr_mlm),
        }
    )

    model_instance_contr_mlm = SpeechEncDecSelfSupervisedModel(cfg=modelConfig_contr_mlm)

    ssl_models = [model_instance_contr_mlm]

    return ssl_models


class TestSSLModel:
    @pytest.mark.unit
    def test_constructor(self, ssl_models):
        for ssl_model in ssl_models:
            ssl_model.train()
            confdict = ssl_model.to_config_dict()
            instance2 = SpeechEncDecSelfSupervisedModel.from_config_dict(confdict)
            assert isinstance(instance2, SpeechEncDecSelfSupervisedModel)

    @pytest.mark.unit
    def test_forward(self, ssl_models):

        for ssl_model in ssl_models:

            ssl_model = ssl_model.eval()

            ssl_model.preprocessor.featurizer.dither = 0.0
            ssl_model.preprocessor.featurizer.pad_to = 16

            input_signal = torch.randn(size=(4, 512))
            length = torch.randint(low=161, high=500, size=[4])

            # batch size 4
            spectrograms, spec_masks, encoded, encoded_len = ssl_model.forward(
                input_signal=input_signal, input_signal_length=length
            )

            loss_value, loss_val_dict = ssl_model.decoder_loss_step(spectrograms, spec_masks, encoded, encoded_len)

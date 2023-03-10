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


import pytorch_lightning as pl
from omegaconf import OmegaConf

from nemo.collections.asr.models import EncDecCTCModelBPE, EncDecRNNTBPEModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

"""
--config-path="conf/conformer/"
--config-name="conformer_ctc_bpe"
model.train_ds.manifest_filepath=/home/samuel/data/ls_100h/tarred_audio_manifest.json
model.train_ds.is_tarred=true
model.train_ds.tarred_audio_filepaths=/home/samuel/data/ls_100h/audio__OP_0..63_CL_.tar
model.train_ds.batch_size=4
++model.train_ds.shuffle_n=128
++model.train_ds.num_workers=2
++model.train_ds.pin_memory=true
++model.train_ds.max_duration=10
++model.validation_ds.manifest_filepath=/home/samuel/data/manifests/ls_test_other_manifest.json
model.validation_ds.batch_size=4
++model.validation_ds.shuffle_n=128
++model.validation_ds.num_workers=2
++trainer.gpus=1
++exp_manager.exp_dir=nemo_experiments/test0
++exp_manager.create_wandb_logger=true
++exp_manager.name="g"
++exp_manager.wandb_logger_kwargs.name=test0
++exp_manager.wandb_logger_kwargs.project=test
++exp_manager.wandb_logger_kwargs.resume=true
++exp_manager.resume_if_exists=true
++exp_manager.resume_ignore_no_checkpoint=true
++exp_manager.checkpoint_callback_params.save_top_k=3
++exp_manager.checkpoint_callback_params.always_save_nemo=true
model.tokenizer.dir=/home/samuel/data/tokenizers/nemo3/tokenizer_spe_unigram_v1024
model.tokenizer.type=bpe
trainer.strategy=dp
trainer.devices=1
++init_from_nemo_model.0.path=/home/samuel/nvidia/ckpt/pt_vp_conf_l.nemo
++init_from_nemo_model.0.exclude=["pre_encode"]
++init_from_nemo_model.1.path=/home/samuel/nvidia/ckpt/dima_conf_2_l_rnnt_n3_300k.nemo
++init_from_nemo_model.1.include=["pre_encode"]
++save_to=/home/samuel/nvidia/ckpt/pt_vp_conf_l_converted.nemo
++model.encoder.d_model=512
++model.encoder.subsampling_conv_channels=256
++model.encoder.subsampling="dw_striding"
++model.encoder.subsampling_factor=8"""

@hydra_runner(config_path="experimental/contextnet_rnnt", config_name="config_rnnt_bpe")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = EncDecCTCModelBPE(cfg=cfg.model, trainer=trainer)
    #asr_model = EncDecRNNTBPEModel(cfg=cfg.model, trainer=trainer)


    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    #asr_model.encoder.copy_local_to_global_qkv()

    #asr_model.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=[64, 64])
    #asr_model.change_attention_model(self_attention_model="rel_pos", att_context_size=[64, 64])

    asr_model.change_conf_layers(new_kernel_size=9)

    asr_model.save_to(cfg.save_to)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter

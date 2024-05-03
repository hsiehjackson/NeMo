# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from omegaconf.omegaconf import OmegaConf
from huggingface_hub import login as hf_login
from nemo.collections.nlp.models.language_modeling.huggingface_model import HuggingfaceModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.utils import logging


@hydra_runner(config_path="conf", config_name="huggingface_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**cfg.trainer)
    trainer.callbacks.extend([
        pl.callbacks.LearningRateMonitor()
    ])
    exp_manager(trainer, cfg.get("exp_manager"))
    
    hf_login(token=cfg.model.get("hf_login_key"))
    del cfg.model.hf_login_key
    model = HuggingfaceModel(cfg.model, trainer)
    trainer.fit(model)


if __name__ == '__main__':
    main()

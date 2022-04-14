import contextlib
import glob
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.models import ASRModel
from nemo.core.config import hydra_runner
from nemo.utils import logging, model_utils

from sklearn.cluster import MiniBatchKMeans, Birch, DBSCAN

from nemo.core.classes.mixins import set_access_cfg, AccessMixin

from tqdm.auto import tqdm

import pickle
@dataclass
class ReduceLabelsConfig:
    in_manifest: str
    out_manifest: str

@hydra_runner(config_name="ReduceLabelsConfig", schema=ReduceLabelsConfig)
def main(cfg: ReduceLabelsConfig) -> ReduceLabelsConfig:
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    with open(cfg.out_manifest, 'w', encoding='utf-8') as f:
        with open(cfg.in_manifest, 'r') as fr:
            for idx, line in enumerate(fr):
                item = json.loads(line)

                new_list = []
                prev_tok = -1
                for tok in item['token_labels']:
                    if tok != prev_tok:
                        prev_tok = tok
                        new_list.append(tok)

                item['token_labels'] = new_list

                f.write(json.dumps(item) + "\n")


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter

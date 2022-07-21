import contextlib
import glob
import json
import os
import pickle
from dataclasses import dataclass
from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from sklearn.cluster import DBSCAN, Birch, MiniBatchKMeans
from tqdm.auto import tqdm

from nemo.collections.asr.models import ASRModel
from nemo.core.classes.mixins import AccessMixin, set_access_cfg
from nemo.core.config import hydra_runner
from nemo.utils import logging, model_utils


from nltk.lm import MLE, Vocabulary
from nltk.lm.preprocessing import padded_everygram_pipeline

import dill as pickle

@dataclass
class PseudoLMConfig:
    in_manifest: str
    out_model: str
    n: int = 5


@hydra_runner(config_name="PseudoLMConfig", schema=PseudoLMConfig)
def main(cfg: PseudoLMConfig) -> PseudoLMConfig:
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    train_data = []

    with open(cfg.in_manifest, 'r') as fr:
        for idx, line in enumerate(fr):
            item = json.loads(line)
            train_data.append(list(map(str, item['token_labels'])))

    train_data, padded_sents = padded_everygram_pipeline(cfg.n, train_data)
    vocab = Vocabulary(padded_sents, unk_cutoff=100)
    print(list(vocab.items))
    input()
    model = MLE(cfg.n, vocabulary=vocab)
    model.fit(train_data, padded_sents)

    with open(cfg.out_model, 'wb') as fout:
        pickle.dump(model, fout)

if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter

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

from nltk.lm import Laplace, Vocabulary
from nltk.lm.preprocessing import padded_everygram_pipeline

import dill as pickle


@dataclass
class PseudoLMConfig:
    in_manifest: str
    out_model: str
    n: int = 4
    unk_cutoff: int = 300
    reduce_ids: bool = True
    max_lines: int = 50000


@hydra_runner(config_name="PseudoLMConfig", schema=PseudoLMConfig)
def main(cfg: PseudoLMConfig) -> PseudoLMConfig:
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    train_data = []

    with open(cfg.in_manifest, 'r') as fr:
        for idx, line in enumerate(fr):
            item = json.loads(line)

            if cfg.reduce_ids:
                token_list = []
                prev_tok = -1
                for tok in item['token_labels']:
                    if tok != prev_tok:
                        prev_tok = tok
                        token_list.append(tok)
            else:
                token_list = item['token_labels']

            train_data.append(list(map(str, token_list)))

            if idx + 1 >= cfg.max_lines:
                break

    #print(train_data[:10])
    print(list(len(i) for i in train_data[:30]))

    train_data, padded_sents = padded_everygram_pipeline(cfg.n, train_data)
    vocab = Vocabulary(padded_sents, unk_cutoff=cfg.unk_cutoff)
    sorted_counts = sorted(list((item, vocab[item]) for item in list(vocab)), key=lambda x: -x[1])
    print(sorted_counts)
    print(len(sorted_counts))
    model = Laplace(cfg.n)
    model.fit(train_data, vocab)

    with open(cfg.out_model, 'wb') as fout:
        pickle.dump(model, fout)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter

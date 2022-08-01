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
    pseudo_lms: List[str]
    in_manifest: str
    reduce_ids: bool = True
    max_lines: int = 50000
    n: int = 4


@hydra_runner(config_name="PseudoLMConfig", schema=PseudoLMConfig)
def main(cfg: PseudoLMConfig) -> PseudoLMConfig:
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    pseudo_lms = []

    for pseudo_lm in cfg.pseudo_lms:
        with open(pseudo_lm, 'rb') as fin:
            lm = pickle.load(fin)
            pseudo_lms.append(lm)

    lan_lm_pp = dict()
    lan_count = dict()

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

            token_list = [list(map(str, token_list))]

            # print(item["audio_filepath"])

            pref_str = "_mnt_disk8b_Data_vox_populi_segmented_"
            lan = item["audio_filepath"][len(pref_str):len(pref_str) + 2]

            if lan not in lan_lm_pp:
                lan_lm_pp[lan] = dict()
                lan_count[lan] = 0

            lan_count[lan] += 1

            pp_bg = 0

            for idx2, pseudo_lm in enumerate(pseudo_lms):
                test_data, _ = padded_everygram_pipeline(cfg.n, token_list)
                for test in test_data:
                    try:
                        pp = pseudo_lm.perplexity(test)
                    except ZeroDivisionError:
                        pp = 0
                    # print(cfg.pseudo_lms[idx], round(pp, 2))

                    if idx2 == 0:
                        pp_bg = pp

                    if idx2 not in lan_lm_pp[lan]:
                        lan_lm_pp[lan][idx2] = 0.
                    lan_lm_pp[lan][idx2] += pp_bg - pp

            # print()
            # input()

            if idx + 1 >= cfg.max_lines:
                break

        for lan, lm_dict in lan_lm_pp.items():
            print(lan)
            for lm_id, pp in lm_dict.items():
                print(cfg.pseudo_lms[lm_id], round(pp / lan_count[lan], 2))
            print()

        print()

        for lm_id, lm in enumerate(cfg.pseudo_lms):
            print(lm)
            scores = dict()
            for lan in lan_count.keys():
                scores[lan] = lan_lm_pp[lan][lm_id] / lan_count[lan]
                # print(lan, lan_count[lan], lan_lm_pp[lan][lm_id] / lan_count[lan])
            scores_sorted = sorted(list(scores.items()), key=lambda x: -x[1])
            for s in scores_sorted:
                print(s)
            print()


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter

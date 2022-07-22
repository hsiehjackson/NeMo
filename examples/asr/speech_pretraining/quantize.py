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


@dataclass
class AccessConfig:
    detach: bool = True
    convert_to_cpu: bool = True


@dataclass
class FeatClusteringConfig:
    # Required configs
    apply_manifests: List[str]  # which manifests to apply clustering to
    apply_is_tarred: Optional[List[bool]]  # which manifests that we are applying to are tarred
    apply_tarred_filepaths: Optional[List[str]]  # lists of filenames for tarred sets
    out_manifests: List[str]  # names of new manifests to write to
    fit_manifest: str  # manifest for fitting the clustering model
    fit_is_tarred: bool
    fit_tarred_filepaths: str

    apply_to_fit: bool = True

    model_path: Optional[str] = None  # Path to a .nemo file
    pretrained_name: Optional[str] = None  # Name of a pretrained model

    load_cluster_model_path: Optional[str] = None  # if already have cluster model
    save_cluster_model_path: Optional[str] = None

    # General configs
    # output_filename: Optional[str] = None
    batch_size: int = 64
    sample_rate: int = 16000

    # Set `cuda` to int to define CUDA device. If 'None', will look for CUDA
    # device anyway, and do inference on CPU only if CUDA device is not found.
    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    amp: bool = False
    # audio_type: str = "wav"

    # clustering params
    cluster_model: str = "KMeans"
    num_feats_for_fit: int = 50000
    n_clusters: Optional[int] = 100

    layer_name: str = "7"

    access: AccessConfig = AccessConfig()

    stride: int = 4
    combine: int = 1

    pad_to: int = 4

    num_samples: int = 50000


def produce_labels(datalayer, in_manifest, out_manifest, asr_model, device, num_samples=None):
    # token_data = []

    label_dict = {}

    for batch in tqdm(datalayer, desc="Getting cluster labels"):

        input_signal = batch[0].to(device)
        input_signal_length = batch[1].to(device)

        quantized_x, quantized_ids, quantized_length = asr_model.quantize(
            input_signal=input_signal, input_signal_length=input_signal_length,
        )

        for j in range(quantized_ids.shape[0]):
            label_dict[int(batch[-1][j])] = quantized_ids[j, : quantized_length[j]]

        del batch

        if num_samples is not None and len(label_dict) >= num_samples:
            break

    ###################

    logging.info(f"Writing labels into file: {out_manifest}")

    # write audio transcriptions
    with open(out_manifest, 'w', encoding='utf-8') as f:
        with open(in_manifest, 'r') as fr:
            for idx, line in enumerate(fr):
                item = json.loads(line)
                if idx in label_dict:
                    item['token_labels'] = list(map(int, label_dict[idx]))
                    f.write(json.dumps(item) + "\n")
                else:
                    print(idx, "not found in label_dict")

    logging.info("Finished writing labels !")

    return


@hydra_runner(config_name="FeatClusteringConfig", schema=FeatClusteringConfig)
def main(cfg: FeatClusteringConfig) -> FeatClusteringConfig:
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    if cfg.model_path is None and cfg.pretrained_name is None:
        raise ValueError("Both cfg.model_path and cfg.pretrained_name cannot be None!")

    # setup GPU
    if cfg.cuda is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(f'cuda:{cfg.cuda}' if cfg.cuda >= 0 else 'cpu')

    trainer = pl.Trainer(accelerator='gpu', strategy='dp', gpus=1)

    # setup model
    if cfg.model_path is not None:
        # restore model from .nemo file path
        model_cfg = ASRModel.restore_from(restore_path=cfg.model_path, return_config=True)
        classpath = model_cfg.target  # original class path
        imported_class = model_utils.import_class_by_path(classpath)  # type: ASRModel
        logging.info(f"Restoring model : {imported_class.__name__}")
        asr_model = imported_class.restore_from(restore_path=cfg.model_path, map_location=device,
                                                trainer=trainer, strict=False)
        model_name = os.path.splitext(os.path.basename(cfg.model_path))[0]
    else:
        # restore model by name
        asr_model = ASRModel.from_pretrained(model_name=cfg.pretrained_name, map_location=device)  # type: ASRModel
        model_name = cfg.pretrained_name

    set_access_cfg(cfg.access)
    asr_model.set_access_enabled(True)
    asr_model.feat_extract_stride = cfg.stride
    asr_model.feat_extract_combine = cfg.combine
    asr_model.preprocessor.featurizer.pad_to = cfg.pad_to

    print(asr_model.access_cfg)

    asr_model = asr_model.eval()

    """
    # get audio filenames
    if cfg.audio_dir is not None:
        filepaths = list(glob.glob(os.path.join(cfg.audio_dir, f"*.{cfg.audio_type}")))
    else:
        # get filenames from manifest
        filepaths = []
        with open(cfg.dataset_manifest, 'r') as f:
            for line in f:
                item = json.loads(line)
                filepaths.append(item['audio_filepath'])
    logging.info(f"\nTranscribing {len(filepaths)} files...\n")
    """

    # setup AMP (optional)
    if cfg.amp and torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
        logging.info("AMP enabled!\n")
        autocast = torch.cuda.amp.autocast
    else:

        @contextlib.contextmanager
        def autocast():
            yield

    # create dataloader like in transcribe
    # ds_cfg = model_cfg.train_ds
    # ds_cfg.manifest_filepath = cfg.dataset_manifest
    # asr_model.setup_training_data(train_data_config=ds_cfg)
    # datalayer = asr_model._train_dl

    ds_cfg = {
        'manifest_filepath': cfg.fit_manifest,
        'sample_rate': cfg.sample_rate,
        'batch_size': cfg.batch_size,
        'trim_silence': False,
        'shuffle': True,
        'num_workers': 16,
        'pin_memory': True,
    }

    ds_cfg = OmegaConf.create(ds_cfg)

    if cfg.fit_is_tarred:
        ds_cfg.is_tarred = True
        ds_cfg.tarred_audio_filepaths = cfg.fit_tarred_filepaths

    ds_cfg.return_sample_id = True
    ds_cfg.min_duration = None
    ds_cfg.max_duration = None
    ds_cfg.shuffle = True

    datalayer = asr_model._setup_dataloader_from_config(ds_cfg)

    # process batches

    asr_model.eval()
    asr_model.to(device)

    if cfg.apply_to_fit:
        print("Producing labels for dataset:", cfg.fit_manifest)
        produce_labels(
            datalayer, cfg.fit_manifest, cfg.out_manifests[0], asr_model, device, num_samples=cfg.num_samples
        )
    # produce labels

    for idx in range(len(cfg.apply_manifests)):
        data_manifest = cfg.apply_manifests[idx]
        if cfg.apply_to_fit:
            out_manifest = cfg.out_manifests[idx + 1]
        else:
            out_manifest = cfg.out_manifests[idx]
        is_tarred = False
        tarred_filepaths = None
        if cfg.apply_is_tarred is not None and cfg.apply_is_tarred[idx]:
            is_tarred = True
            tarred_filepaths = cfg.apply_tarred_filepaths[idx]

        ds_cfg.manifest_filepath = data_manifest
        ds_cfg.is_tarred = is_tarred
        ds_cfg.tarred_audio_filepaths = tarred_filepaths

        datalayer = asr_model._setup_dataloader_from_config(ds_cfg)

        print("Producing labels for dataset:", data_manifest)
        produce_labels(datalayer, data_manifest, out_manifest, asr_model, device, num_samples=cfg.num_samples)

    return cfg


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter

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
from dataclasses import dataclass
from typing import Optional

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.metrics.rnnt_wer import RNNTDecodingConfig
from nemo.collections.asr.models import ASRModel
from nemo.core.config import hydra_runner
from nemo.utils import logging, model_utils

from sklearn.cluster import MiniBatchKMeans

import nemo.collections.asr.parts.mixins as mixins


"""
Transcribe audio file on a single CPU/GPU. Useful for transcription of moderate amounts of audio data.

# Arguments
  model_path: path to .nemo ASR checkpoint
  pretrained_name: name of pretrained ASR model (from NGC registry)
  audio_dir: path to directory with audio files
  dataset_manifest: path to dataset JSON manifest file (in NeMo format)
  
  output_filename: Output filename where the transcriptions will be written
  batch_size: batch size during inference
  
  cuda: Optional int to enable or disable execution of model on certain CUDA device.
  amp: Bool to decide if Automatic Mixed Precision should be used during inference
  audio_type: Str filetype of the audio. Supported = wav, flac, mp3
  
  overwrite_transcripts: Bool which when set allowes repeated transcriptions to overwrite previous results.
  
  rnnt_decoding: Decoding sub-config for RNNT. Refer to documentation for specific values.

# Usage
ASR model can be specified by either "model_path" or "pretrained_name".
Data for transcription can be defined with either "audio_dir" or "dataset_manifest".
Results are returned in a JSON manifest file.

python transcribe_speech.py \
    model_path=null \
    pretrained_name=null \
    audio_dir="" \
    dataset_manifest="" \
    output_filename="" \
    batch_size=32 \
    cuda=0 \
    amp=True
"""

@dataclass
class AccessConfig:
    access_all_intermediate: bool = True

@dataclass
class FeatClusteringConfig:
    # Required configs
    model_path: Optional[str] = None  # Path to a .nemo file
    pretrained_name: Optional[str] = None  # Name of a pretrained model
    dataset_manifest: Optional[str] = None  # Path to dataset's JSON manifest

    cluster_model_path: Optional[str] = None # if already have cluster model

    # General configs
    output_filename: Optional[str] = None
    batch_size: int = 32
    sample_rate: int = 16000

    # Set `cuda` to int to define CUDA device. If 'None', will look for CUDA
    # device anyway, and do inference on CPU only if CUDA device is not found.
    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    amp: bool = False
    #audio_type: str = "wav"

    #clustering params
    num_feats_for_fit: int = 50000
    n_clusters: int = 100

    layer_name: str = "7"

    access: AccessConfig = AccessConfig()

@hydra_runner(config_name="FeatClusteringConfig", schema=FeatClusteringConfig)
def main(cfg: TranscriptionConfig) -> TranscriptionConfig:
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    if cfg.model_path is None and cfg.pretrained_name is None:
        raise ValueError("Both cfg.model_path and cfg.pretrained_name cannot be None!")
    if cfg.audio_dir is None and cfg.dataset_manifest is None:
        raise ValueError("Both cfg.audio_dir and cfg.dataset_manifest cannot be None!")

    # setup GPU
    if cfg.cuda is None:
        if torch.cuda.is_available():
            cfg.cuda = 0  # use 0th CUDA device
        else:
            cfg.cuda = -1  # use CPU

    device = torch.device(f'cuda:{cfg.cuda}' if cfg.cuda >= 0 else 'cpu')

    mixins.set_access_cfg(cfg.access)

    # setup model
    if cfg.model_path is not None:
        # restore model from .nemo file path
        model_cfg = ASRModel.restore_from(restore_path=cfg.model_path, return_config=True)
        classpath = model_cfg.target  # original class path
        imported_class = model_utils.import_class_by_path(classpath)  # type: ASRModel
        logging.info(f"Restoring model : {imported_class.__name__}")
        asr_model = imported_class.restore_from(restore_path=cfg.model_path, map_location=device)  # type: ASRModel
        model_name = os.path.splitext(os.path.basename(cfg.model_path))[0]
    else:
        # restore model by name
        asr_model = ASRModel.from_pretrained(model_name=cfg.pretrained_name, map_location=device)  # type: ASRModel
        model_name = cfg.pretrained_name

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


    #create dataloader like in transcribe
    #ds_cfg = model_cfg.train_ds
    #ds_cfg.manifest_filepath = cfg.dataset_manifest
    #asr_model.setup_training_data(train_data_config=ds_cfg)
    #datalayer = asr_model._train_dl

    ds_cfg = {
        'manifest_filepath': cfg.dataset_manifest,
        'sample_rate': cfg.sample_rate,
        'batch_size': cfg.batch_size,
        'trim_silence': False,
        'shuffle': False,
        'num_workers': 16,
        'pin_memory': True,
    }

    ds_cfg = OmegaConf.create(ds_cfg)

    ds_cfg.return_sample_id = True
    ds_cfg.min_duration = None
    ds_cfg.max_duration = None
    ds_cfg.shuffle = False

    datalayer = asr_model._setup_dataloader_from_config(ds_cfg)

    #process batches

    asr_model.eval()
    asr_model.to(device)
    feats_for_fit = []
    total_feats = 0

    #print(datalayer)
    print(len(datalayer.dataset))

    #indexes = set()

    for batch in tqdm(datalayer, desc="Obtaining features for fit"):
        #for i in batch[-1]:
        #    indexes.add(int(i))

        input_signal = batch[0].to(device)
        input_signal_length = batch[1].to(device)

        feats, feat_lens = asr_model.get_feats(
            input_signal=input_signal,
            input_signal_length=input_signal_length,
            layer_name=cluster_cfg.layer_name
        )

        feats = feats.reshape(-1, feats.shape[-1])
        total_feats += feats.shape[0]

        feats_for_fit.append(feats)

        del batch

        if total_feats > cfg.num_feats_for_fit:
            break

    feats_for_fit = torch.cat(feats_for_fit, dim=0).cpu()

    #do clustering

    cluster_model = MiniBatchKMeans(n_clusters=cfg.n_clusters,
                                    max_iter=10000,
                                    tol=0.0,
                                    max_no_improvement=200,
                                    n_init=20,
                                    reassignment_ratio=0.01,
                                    batch_size=10000,
                                    verbose=True)

    cluster_model.fit(feats_for_fit)
    inertia = -cluster_model.score(feats_for_fit) / len(feats_for_fit)
    print("total intertia: %.5f", inertia)
    print("finished successfully")


    #produce labels

    label_dict = {}
    dur_dict = {}

    cluster_labels = []
    for batch in tqdm(datalayer, desc="Getting cluster labels"):

        input_signal = batch[0].to(device)
        input_signal_length = batch[1].to(device)

        feats, feat_lens = asr_model.get_feats(
            input_signal=input_signal,
            input_signal_length=input_signal_length,
            layer_name=cluster_cfg.layer_name
        )

        orig_bs = feats.shape[0]

        feats = feats.reshape(-1, feats.shape[-1])
        cur_labels = cluster_model.predict(feats.cpu())
        cur_labels = cur_labels.reshape(orig_bs, -1)

        for j in range(cur_labels.shape[0]):
            label_dict[int(batch[-1][j])] = cur_labels[j, :feat_lens[j]]
            dur_dict[int(batch[-1][j])] = int(input_signal_length[j])

        del batch



    ###################

    logging.info(f"Writing labels into file: {cfg.output_filename}")

    # write audio transcriptions
    with open(cfg.output_filename, 'w', encoding='utf-8') as f:
        with open(cfg.dataset_manifest, 'r') as fr:
                for idx, line in enumerate(fr):
                    item = json.loads(line)
                    item['labels'] = list(map(int, label_dict[idx]))
                    f.write(json.dumps(item) + "\n")

    logging.info("Finished writing labels !")

    return cfg


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter

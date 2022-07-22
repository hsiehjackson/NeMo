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

"""
This is a helper script to extract speaker embeddings based on manifest file
Usage:
python extract_speaker_embeddings.py --manifest=/path/to/manifest/file' 
--model_path='/path/to/.nemo/file'(optional)
--embedding_dir='/path/to/embedding/directory'

Args:
--manifest: path to manifest file containing audio_file paths for which embeddings need to be extracted
--model_path(optional): path to .nemo speaker verification model file to extract embeddings, if not passed SpeakerNet-M model would 
    be downloaded from NGC and used to extract embeddings
--embeddings_dir(optional): path to directory where embeddings need to stored default:'./'


"""

import json
import os
import pickle as pkl
from argparse import ArgumentParser

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from nemo.collections.asr.models.ssl_models import SpeechEncDecSelfSupervisedModel
from nemo.collections.asr.parts.utils.speaker_utils import embedding_normalize
from nemo.utils import logging

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


def get_embeddings(ssl_model, manifest_file, batch_size=1, embedding_dir='./', layer=0, device='cuda'):
    test_config = OmegaConf.create(
        dict(manifest_filepath=manifest_file, sample_rate=16000, labels=None, batch_size=batch_size, shuffle=False,)
    )

    ssl_model.setup_validation_data(test_config)
    ssl_model = ssl_model.to(device)
    ssl_model.eval()

    all_embs = []
    out_embeddings = {}

    ssl_model.apply_masking = False
    if layer:
        ssl_model.access_cfg['save_encoder_tensors'] = True
        ssl_model.set_access_enabled(access_enabled=True)

    for test_batch in tqdm(ssl_model._validation_dl):
        test_batch = [x.to(device) for x in test_batch]
        signal, signal_len, targets, target_lengths = test_batch
        if layer == -1:
            with autocast():
                spects, spec_len = ssl_model.preprocessor(input_signal=signal, length=signal_len)
                targets = spects.transpose(-2, -1)
                targets = targets.reshape(
                    targets.shape[0],
                    targets.shape[1] // ssl_model.decoder_losses.contrastive.loss.combine_time_steps,
                    -1,
                )
                embs, _, _, ids = ssl_model.decoder_losses.contrastive.loss.quantizer(targets, return_ids=True)
                all_embs.extend(embs.cpu().detach().numpy())
            del test_batch
            continue
        with autocast():
            spectrograms, spec_masks, encoded, encoded_len = ssl_model(
                input_signal=signal, input_signal_length=signal_len
            )
            if layer:
                registry = ssl_model.get_module_registry(ssl_model.encoder)
                embs = registry[f'layers.{layer-1}']['encoder'][-1]
                embs = embs.transpose(-2, -1)
            embs = torch.mean(encoded, dim=-1)
            emb_shape = embs.shape[-1]
            embs = embs.view(-1, emb_shape)
            all_embs.extend(embs.cpu().detach().numpy())
        del test_batch

    all_embs = np.asarray(all_embs)
    all_embs = embedding_normalize(all_embs) if layer != -1 else all_embs
    with open(manifest_file, 'r', encoding='utf-8') as manifest:
        for i, line in enumerate(manifest.readlines()):
            line = line.strip()
            dic = json.loads(line)
            uniq_name = '@'.join(dic['audio_filepath'].split('/')[-3:])
            out_embeddings[uniq_name] = all_embs[i]

    embedding_dir = os.path.join(embedding_dir, 'embeddings')
    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir, exist_ok=True)

    prefix = manifest_file.split('/')[-1].rsplit('.', 1)[-2]

    name = os.path.join(embedding_dir, prefix)
    embeddings_file = name + '_embeddings.pkl'
    pkl.dump(out_embeddings, open(embeddings_file, 'wb'))
    logging.info("Saved embedding files to {}".format(embedding_dir))


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--manifest", type=str, required=True, help="Path to manifest file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default='titanet_large',
        required=False,
        help="path to .nemo speaker verification model file to extract embeddings, if not passed SpeakerNet-M model would be downloaded from NGC and used to extract embeddings",
    )
    parser.add_argument(
        "--embedding_dir",
        type=str,
        default='./',
        required=False,
        help="path to directory where embeddings need to stored default:'./'",
    )
    parser.add_argument(
        "--layer", type=int, default=0, required=False, help="which layer of embeddings to extract",
    )
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    if args.model_path.endswith('.nemo'):
        logging.info(f"Using local speaker model from {args.model_path}")
        ssl_model = SpeechEncDecSelfSupervisedModel.restore_from(restore_path=args.model_path)
    elif args.model_path.endswith('.ckpt'):
        ssl_model = SpeechEncDecSelfSupervisedModel.load_from_checkpoint(checkpoint_path=args.model_path)
    else:
        ssl_model = SpeechEncDecSelfSupervisedModel.from_pretrained(model_name="titanet_large")
        logging.info(f"using pretrained titanet_large speaker model from NGC")

    device = 'cuda'
    if not torch.cuda.is_available():
        device = 'cpu'
        logging.warning("Running model on CPU, for faster performance it is adviced to use atleast one NVIDIA GPUs")

    get_embeddings(
        ssl_model, args.manifest, batch_size=1, embedding_dir=args.embedding_dir, layer=args.layer, device=device
    )


if __name__ == '__main__':
    main()
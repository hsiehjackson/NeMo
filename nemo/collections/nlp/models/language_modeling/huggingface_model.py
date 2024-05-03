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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM

from nemo.core.classes import ModelPT, Exportable
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.common.losses import SmoothedCrossEntropyLoss
from nemo.utils import logging, model_utils

try:
    from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
    from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
    from megatron.core.datasets.bert_dataset import BERTMaskedWordPieceDataset, BERTMaskedWordPieceDatasetConfig
    from megatron.core.datasets.t5_dataset import T5MaskedWordPieceDataset, T5MaskedWordPieceDatasetConfig
    HAVE_MEGATRON_CORE = True
    
except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False

__all__ = ["HuggingfaceModel"]
hf_model_class_mapping = {
    'AutoModelForCausalLM': AutoModelForCausalLM,
    'AutoModelForSeq2SeqLM': AutoModelForSeq2SeqLM,
    'AutoModelForMaskedLM': AutoModelForMaskedLM
}

class HuggingfaceModel(ModelPT, Exportable):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)
        super().__init__(cfg=cfg, trainer=trainer)
        
        self.tokenizer = self.setup_tokenizer()
        self.model, self.model_config = self.setup_model()
        logging.info(f'Number of model parameters: {self.count_parameters():.2e}.')
        
        self.loss_func = SmoothedCrossEntropyLoss(
            pad_id=self.tokenizer.pad_id, 
            label_smoothing=0.0,
            per_token_reduction=True,
        )
        
    def setup(self,stage):
        # PTL
        dataset = self.setup_dataset()
        self._train_ds = dataset[0]
        self._train_dl = self.setup_dataloader(
            type='train',
            dataset=self._train_ds, 
            consumed_samples=self.compute_consumed_samples(),
        )
        self._validation_ds = dataset[1]
        self._validation_dl = self.setup_dataloader(
            type='validation',
            dataset=self._validation_ds, 
            consumed_samples=0,
        )
        
    def forward(
        self, 
        input_ids: torch.Tensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):
        # PTL
        model_inputs = {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "head_mask": head_mask,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "decoder_inputs_embeds": decoder_inputs_embeds,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "labels": labels,
            "use_cache": use_cache,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
        }
        model_inputs = {k: v for k, v in model_inputs.items() if v is not None}
        return self.model(**model_inputs)
    
    def generate(self, input: str, dec_input: Optional[str]=None, **generation_kwargs):
        generation_kwargs['input_ids'] = torch.LongTensor([self.tokenizer.text_to_ids(input)]).to(self.model.device)
        input_length = generation_kwargs['input_ids'].shape[1]
        
        if dec_input:
            generation_kwargs['decoder_input_ids'] = torch.LongTensor([self.tokenizer.text_to_ids(dec_input)]).to(self.model.device)
            input_length = generation_kwargs['decoder_input_ids'].shape[1]
            
        output_id = self.model.generate(**generation_kwargs)[0].tolist()
        output = self.tokenizer.ids_to_text(output_id[input_length:])
        return output

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters())

    def training_step(self, batch, batch_idx):
        losses = self.forward_one_step(batch)
        self.log('train_loss', losses, prog_bar=True, batch_size=1, sync_dist=True)
        return losses

    def validation_step(self, batch, batch_idx):
        losses = self.forward_one_step(batch)
        self.validation_step_outputs.append(losses)
        return losses

    def forward_one_step(self, batch):
        model_inputs = {
            "input_ids": batch.get('tokens', batch.get('text', batch.get('text_enc'))),
            "attention_mask": batch.get('attention_mask', batch.get('padding_mask', batch.get('enc_mask'))),
            "token_type_ids": batch.get('types'),
            "position_ids": batch.get('position_ids'),
            "decoder_input_ids": batch.get('text_dec'),
            "decoder_attention_mask": batch.get('dec_mask'),
        }
        outputs = self(**model_inputs)
        logits = outputs.get("logits")
        log_probs = torch.log_softmax(logits, dim=-1)
        labels = batch.get("labels")
        loss_mask = batch.get("loss_mask")
        losses = self.loss_func(
            log_probs=log_probs, 
            labels=labels, 
            output_mask=loss_mask,
        )
        return losses
        
    def on_validation_epoch_start(self):
        # PTL
        self.log('consumed_samples', self.compute_consumed_samples(), prog_bar=True, rank_zero_only=True, batch_size=1)
        
    def on_validation_epoch_end(self):
        # PTL
        averaged_loss = torch.stack(self.validation_step_outputs).mean()
        self.log('val_loss', averaged_loss, prog_bar=True, batch_size=1, sync_dist=True)
        self.validation_step_outputs.clear()  # free memory
        return averaged_loss

    def setup_model(self):
        assert self.cfg.get('hf_model_class') in hf_model_class_mapping, \
        f'`hf_model_class` should be in {hf_model_class_mapping.keys()}.'
        
        model_class = hf_model_class_mapping[self.cfg.get('hf_model_class')]
        model_config = AutoConfig.from_pretrained(self.cfg.get('hf_model_name_or_path'))
        model_config.update({
            'vocab_size': self.tokenizer.vocab_size,
            'bos_token_id': self.tokenizer.bos_id,
            'eos_token_id': self.tokenizer.eos_id,
            'pad_token_id': self.tokenizer.pad_id,
        })
        if self.cfg.get('hf_config'):
            model_config.update(OmegaConf.to_container(self.cfg.get('hf_config')))
            
        model = model_class.from_config(model_config)

        # model = model_class.from_pretrained(self.cfg.get('hf_model_name_or_path'))
        # model_config = model.config

        return model, model_config.to_dict()
    
    def setup_tokenizer(self):
        tokenizer = get_nmt_tokenizer(
            library=self.cfg.tokenizer.library,
            model_name=self.cfg.tokenizer.name,
            tokenizer_model= self.register_artifact("tokenizer.model", self.cfg.tokenizer.get('model')),
            vocab_file=self.register_artifact("tokenizer.vocab_file", self.cfg.tokenizer.get('vocab_file')),
            merges_file=self.register_artifact("tokenizer.merge_file", self.cfg.tokenizer.get('merge_file')),
            use_fast=self.cfg.tokenizer.get('use_fast', False),
            delimiter=self.cfg.tokenizer.get('delimiter', None),
            special_tokens=self.cfg.tokenizer.get('special_tokens', None),
            trust_remote_code=self.cfg.tokenizer.get('trust_remote_code', False),
            legacy=self.cfg.tokenizer.get('sentencepiece_legacy', False),
        )
        
        return tokenizer


    def setup_training_data(self, cfg):
        # ModelPT
        pass 
        
    def setup_validation_data(self, cfg):
        # ModelPT
        pass

    def list_available_models(self,):
        # ModelPT
        return None

    def setup_dataset(self,):        
        max_train_steps = self.trainer.max_steps
        max_valid_steps = (max_train_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches
        global_batch_size = self.trainer.accumulate_grad_batches \
                   * self.cfg.micro_batch_size \
                   * torch.distributed.get_world_size()
        total_num_samples = [
            max_train_steps * global_batch_size,
            max_valid_steps * global_batch_size,
        ]

        default_config = {
            "random_seed": self.cfg.seed,
            "blend_per_split": [self.cfg.data.data_prefix.get('train'), self.cfg.data.data_prefix.get('validation')] \
                                if isinstance(self.cfg.data.data_prefix, DictConfig) else None,
            "blend": self.cfg.data.get('data_prefix'),
            "split": self.cfg.data.get('splits_string'),
            "sequence_length": self.model_config.get('n_positions') or self.model_config.get('max_position_embeddings'),
            "path_to_cache": self.cfg.data.index_mapping_dir,
            "mmap_bin_files": self.cfg.data.get("mmap_bin_files", True),
            "tokenizer": self.tokenizer,
        }
        
        if self.cfg.data.get('dataset_type') == 'gpt':
            dataset_type = GPTDataset
            dataset_config = GPTDatasetConfig(
                reset_position_ids=self.cfg.data.get('gpt_reset_position_ids', False),
                reset_attention_mask=self.cfg.data.get('gpt_reset_attention_mask', False),
                eod_mask_loss=self.cfg.data.get('gpt_eod_mask_loss', False),
                create_attention_mask=self.cfg.data.get('gpt_create_attention_mask', False),
                **default_config
            )
        elif self.cfg.data.get('dataset_type') == 't5':
            dataset_type = T5MaskedWordPieceDataset
            dataset_config = T5MaskedWordPieceDatasetConfig(
                masking_probability=self.cfg.data.get('t5_masked_lm_prob', 0.15),
                short_sequence_probability=self.cfg.data.get('t5_short_seq_prob', 0.0),
                masking_max_ngram=self.cfg.data.get('t5_max_ngram_size', 10),
                masking_do_full_word=self.cfg.data.get('t5_whole_word_masking', True),
                masking_do_permutation=self.cfg.data.get('t5_permutation', False),
                masking_use_longer_ngrams=self.cfg.data.get('t5_favor_longer_ngrams', False),
                masking_use_geometric_distribution=self.cfg.data.get('t5_geometric_dist', True),
                **default_config
            )
        elif self.cfg.data.get('dataset_type') == 'bert':
            dataset_type = BERTMaskedWordPieceDataset
            dataset_config = BERTMaskedWordPieceDatasetConfig(
                masking_probability=self.cfg.data.get('bert_masked_lm_prob', 0.15),
                short_sequence_probability=self.cfg.data.get('bert_short_seq_prob', 0.1),
                **default_config
            )
        else:
            raise NotImplementedError

        return BlendedMegatronDatasetBuilder(
            dataset_type, 
            sizes=total_num_samples, 
            is_built_on_rank=lambda: True, 
            config=dataset_config,
        ).build()
    
    def setup_dataloader(self, type, dataset, consumed_samples):
        """Buld dataloader given an input dataset."""
        logging.info(f'Setting up {type} dataloader with length: {len(dataset)} and consumed samples: {consumed_samples}')
        
        if self.cfg.data.get('dataloader_type') == 'single':
            batch_sampler = MegatronPretrainingSampler(
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=self.cfg.micro_batch_size,
                global_batch_size=None,
                data_parallel_rank=torch.distributed.get_rank(),
                data_parallel_size=torch.distributed.get_world_size(),
                pad_samples_to_global_batch_size=False,
                drop_last=self.cfg.data.get('drop_last', True),
            )
        elif self.cfg.data.get('dataloader_type') == 'cyclic':
            batch_sampler = MegatronPretrainingRandomSampler(
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=self.cfg.micro_batch_size,
                global_batch_size=None,
                data_parallel_rank=torch.distributed.get_rank(),
                data_parallel_size=torch.distributed.get_world_size(),
                pad_samples_to_global_batch_size=False,
                drop_last=self.cfg.data.get('drop_last', True),
            )
        else:
            raise NotImplementedError('cfg.data.dataloader_type must be "single" or "cyclic"')

        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            persistent_workers=True if self.cfg.data.num_workers > 0 else False,
        )

    def compute_consumed_samples(self):
        return int(self.trainer.global_step \
                   * self.cfg.micro_batch_size \
                   * torch.distributed.get_world_size() \
                   * self.trainer.accumulate_grad_batches)
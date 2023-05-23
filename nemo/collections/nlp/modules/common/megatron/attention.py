# coding=utf-8
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
import math

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat

from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import AdapterName, InfusedAdapterConfig
from nemo.collections.nlp.modules.common.megatron.fused_softmax import MatchedScaleMaskSoftmax
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.rotary_pos_embedding import apply_rotary_pos_emb
from nemo.collections.nlp.modules.common.megatron.xpos_relative_position import XPOS
from nemo.collections.nlp.modules.common.megatron.sandwich_relative_position import sandwich_pos_bias
from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults, attention_mask_func
from nemo.core import adapter_mixins
from nemo.utils import logging

import math
from functools import lru_cache
from typing import List, Tuple

from nemo.collections.nlp.modules.common.megatron.alibi_relative_position_embedding import (
    ALiBiRelativePositionEmbedding,
)
from nemo.collections.nlp.modules.common.megatron.layer_type import LayerType

from nemo.utils import avoid_float16_autocast_context

try:
    from apex.transformer import parallel_state, tensor_parallel
    from apex.transformer.enums import AttnMaskType, AttnType
    from apex.transformer.utils import divide as safe_divide
    from apex._autocast_utils import _cast_if_autocast_enabled

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

    # fake missing classes with None attributes
    ModelType = AttnMaskType = AttnType = LayerType = ApexGuardDefaults()

try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func
    from flash_attn.flash_attn_triton import flash_attn_func
    from flash_attn.bert_padding import unpad_input, pad_input

except (ImportError, ModuleNotFoundError):
    logging.warning(
        "flash_attn was not found. Please see the installation instructions: https://github.com/HazyResearch/flash-attention."
        "If you use flash_attn with triton. Please see the installation instructions: https://github.com/openai/triton/."
    )
    flash_attn_unpadded_func, flash_attn_func = None, None
    unpad_input, pad_input = None, None

""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
"""


def _pad_to_multiple(x: torch.Tensor, block_len: int, dim: int, pad_value: int = 0) -> torch.Tensor:
    """Pad a tensor so that a sequence length will be a multiple of `block_len`"""
    pad_len = -x.shape[dim] % block_len
    # Handle cases when an empty input sequence is given
    if not all(x.shape):
        new_shape = list(x.shape)
        new_shape[dim] += pad_len
        return torch.zeros(new_shape, dtype=x.dtype)

    pad = [(0, 0)] * x.ndim
    pad[dim] = (0, pad_len)
    pad = sum(pad[::-1], ())
    x = nn.functional.pad(x, pad=pad, mode="constant", value=pad_value)
    return x


def _split_into_blocks(x: torch.Tensor, block_len: int, dim: int) -> torch.Tensor:
    """Split an input tensor into blocks of a given `block_len` along the given `dim`. If the dimension length
    is not a multiple of `block_len`, it will be padded first with selected `pad_value`.
    """
    # pad tensor to multiple of block_len
    if x.shape[dim] % block_len != 0:
        x = _pad_to_multiple(x, block_len, dim, pad_value=0)
    num_blocks = x.shape[dim] // block_len
    output_shape = x.shape[:dim] + (num_blocks, block_len) + x.shape[(dim + 1) :]
    # If 0 is in output_shape, we cannot apply reshape because of incompatibility with ONNX conversion
    if 0 in output_shape:
        return torch.empty(output_shape, dtype=x.dtype, device=x.device)
    return x.reshape(output_shape)


def _concatenate_3_blocks(x: torch.Tensor, block_dim: int, sequence_dim: int, pad_value: int = 0) -> torch.Tensor:
    """Concatenate three consecutive blocks for each input block for local attentiont.

    For more information, see: https://arxiv.org/pdf/2112.07916.pdf.
    """
    num_blocks = x.shape[block_dim]

    pad = [(0, 0)] * x.ndim
    pad[block_dim] = (1, 1)
    pad = sum(pad[::-1], ())
    # [batch_size, num_blocks, block_len] -> [batch_size, num_blocks + 2, block_len]
    x = nn.functional.pad(x, pad=pad, mode="constant", value=pad_value)

    blocks_list: List[torch.Tensor] = []
    for i in range(3):
        # We use indexing approach here:
        # https://numpy.org/doc/stable/user/basics.indexing.html#dealing-with-variable-numbers-of-indices-within-programs
        indices = [slice(0, None)] * x.ndim
        indices[block_dim] = slice(i, i + num_blocks)
        indices = tuple(indices)
        blocks_list.append(x[indices])
    # [batch_size, num_blocks, 3 * block_len, ...]
    return torch.cat(blocks_list, dim=sequence_dim)


def _make_3block_relative_position_ids(block_len: int) -> torch.Tensor:
    """Makes 3-blocked relative position ids for local attention."""
    position_ids = torch.arange(3 * block_len, dtype=torch.int32)
    center_position_ids = position_ids[block_len:-block_len]
    # [block_len, 3 * block_len]
    relative_position_ids = position_ids.unsqueeze(0) - center_position_ids.unsqueeze(1)
    return relative_position_ids


def _mask_local_attention_mask(local_attention_mask: torch.Tensor, block_len: int) -> torch.Tensor:
    """Mask local attention mask to enforce that tokens are not allowed to attend tokens farther than ``local_radius."""
    relative_position_ids = _make_3block_relative_position_ids(block_len)
    locality_mask = torch.abs(relative_position_ids) < block_len
    locality_mask = locality_mask[None, None, :, :]
    locality_mask = locality_mask.to(local_attention_mask.device)
    return torch.logical_and(local_attention_mask, locality_mask)


def _get_local_attention_mask(attention_mask: torch.Tensor, block_len: int, device: torch.device) -> torch.Tensor:
    """Prepare attention mask to be applied for a local attention."""
    # [batch_size, num_blocks, block_len]
    _blocked_attention_mask = _split_into_blocks(attention_mask, block_len, dim=1)
    # [batch_size, num_block, 3 * block_len]
    _3blocked_attention_mask = _concatenate_3_blocks(_blocked_attention_mask, block_dim=1, sequence_dim=2)

    _blocked_attention_mask = _blocked_attention_mask.unsqueeze(-1)
    _3blocked_attention_mask = _3blocked_attention_mask.unsqueeze(-2)
    # [batch_size, num_block, block_len, 3 * block_len]
    local_attention_mask = torch.logical_and(_blocked_attention_mask, _3blocked_attention_mask)
    local_attention_mask = _mask_local_attention_mask(local_attention_mask, block_len)
    # [batch_size, 1, num_block, block_len, 3 * block_len]
    return local_attention_mask.unsqueeze(1).to(device)


class LongT5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the LongT5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # LongT5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class ParallelAttention(MegatronModule, adapter_mixins.AdapterModuleMixin):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        init_method,
        output_layer_init_method,
        layer_number,
        num_attention_heads,
        hidden_size,
        attention_type=AttnType.self_attn,
        attn_mask_type=AttnMaskType.padding,
        precision=16,
        apply_query_key_layer_scaling=True,
        kv_channels=None,
        use_cpu_initialization=False,
        masked_softmax_fusion=True,
        attention_dropout=0.1,
        layer_type=None,
        megatron_legacy=False,
        bias=True,
        headscale=False,
        position_embedding_type='learned_absolute',
        multi_query_attention=False,
        activations_checkpoint_granularity=None,
        sequence_parallel=False,
        gradient_accumulation_fusion=False,
        normalize_attention_scores=True,
        use_long_attention=False,
        local_context=128,
        global_tokens=1024,
        global_tokens_spacing=16,
        global_attn_separate=True,
        transient_global_tokens=False,  # tmp
        global_token_mode="equal_spacing",
        use_flash_attention=False,
    ):
        super(ParallelAttention, self).__init__()

        self.use_long_attention = use_long_attention
        self.local_context = local_context
        self.global_tokens = global_tokens
        self.global_tokens_spacing = global_tokens_spacing
        self.global_attn_separate = False
        self.transient_global_tokens = transient_global_tokens
        if self.transient_global_tokens:
            global_token_mode = "transient"
        self.global_token_mode = global_token_mode

        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.normalize_attention_scores = normalize_attention_scores
        self.position_embedding_type = position_embedding_type
        self.multi_query_attention = multi_query_attention

        self.megatron_legacy = megatron_legacy

        self.set_accepted_adapter_types([InfusedAdapterConfig._target_])

        if kv_channels is None:
            assert (
                hidden_size % num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = hidden_size // num_attention_heads
        projection_size = kv_channels * num_attention_heads

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = safe_divide(projection_size, num_attention_heads)
        self.num_attention_heads_per_partition = safe_divide(num_attention_heads, world_size)
        self.num_attention_heads_partition_offset = (
            self.num_attention_heads_per_partition * parallel_state.get_tensor_model_parallel_rank()
        )

        no_async_tensor_model_parallel_allreduce = (
            parallel_state.get_tensor_model_parallel_world_size() == 1 or sequence_parallel
        )

        # Strided linear layer.
        if attention_type == AttnType.self_attn:
            out_size = 3 * projection_size
            if self.use_long_attention and self.global_tokens > 0 and self.global_attn_separate:
                out_size *= 2
            self.query_key_value = tensor_parallel.ColumnParallelLinear(
                hidden_size,
                out_size,
                gather_output=False,
                init_method=init_method,
                use_cpu_initialization=use_cpu_initialization,
                bias=bias,
                sequence_parallel_enabled=sequence_parallel,
                no_async_tensor_model_parallel_allreduce=no_async_tensor_model_parallel_allreduce,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
            )
        else:
            assert attention_type == AttnType.cross_attn
            self.query = tensor_parallel.ColumnParallelLinear(
                hidden_size,
                projection_size,
                gather_output=False,
                init_method=init_method,
                bias=bias,
                sequence_parallel_enabled=sequence_parallel,
                no_async_tensor_model_parallel_allreduce=no_async_tensor_model_parallel_allreduce,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
            )
            # apex/transformer/tensor_parallel/layers.py will have non-contiguous grad_output

            if self.multi_query_attention:
                kv_proj = kv_channels * 2
            else:
                kv_proj = projection_size * 2

            self.key_value = tensor_parallel.ColumnParallelLinear(
                hidden_size,
                kv_proj,
                gather_output=False,
                init_method=init_method,
                bias=bias,
                sequence_parallel_enabled=sequence_parallel,
                no_async_tensor_model_parallel_allreduce=no_async_tensor_model_parallel_allreduce,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
            )

        if self.global_token_mode == "transient":
            self.transient_norm = LongT5LayerNorm(hidden_size)

        self.core_attention = CoreAttention(
            layer_number=self.layer_number,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            attention_type=self.attention_type,
            attn_mask_type=self.attn_mask_type,
            precision=precision,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            kv_channels=kv_channels,
            masked_softmax_fusion=masked_softmax_fusion,
            attention_dropout=attention_dropout,
            multi_query_attention=multi_query_attention,
            sequence_parallel=sequence_parallel,
            normalize_attention_scores=normalize_attention_scores,
            position_embedding_type=position_embedding_type,
            use_long_attention=use_long_attention,
            local_context=local_context,
            global_tokens=global_tokens,
            global_tokens_spacing=global_tokens_spacing,
            global_attn_separate=False,
            transient_global_tokens=self.global_token_mode == "transient",
            use_flash_attention=use_flash_attention,
        )

        # Output.
        self.dense = tensor_parallel.RowParallelLinear(
            projection_size,
            hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            use_cpu_initialization=use_cpu_initialization,
            bias=bias,
            sequence_parallel_enabled=sequence_parallel,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
        )

        self.headscale = headscale
        if headscale:
            self.head_scale_tensor = torch.nn.Parameter(
                torch.ones(1, self.num_attention_heads_per_partition, 1, 1), requires_grad=True
            )

        # Inference key-value memory
        self.inference_key_memory = None
        self.inference_value_memory = None
        self.inference_current_sequence_len = 0

        # relative position embedding
        self.layer_type = layer_type
        
        if position_embedding_type.lower() == 'xpos':
            self.xpos = XPOS(kv_channels)

    def _checkpointed_attention_forward(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        rotary_pos_emb=None,
        relative_position_bias=None,
        headscale_tensor=None,
    ):
        """Forward method with activation checkpointing."""

        def custom_forward(*inputs):
            if len(inputs) == 7:
                query_layer = inputs[0]
                key_layer = inputs[1]
                value_layer = inputs[2]
                attention_mask = inputs[3]
                rotary_pos_emb = inputs[4]
                relative_position_bias = inputs[5]
                headscale_tensor = inputs[6]
            elif len(inputs) == 8:
                query_layer = inputs[0]
                key_layer = inputs[1]
                value_layer = inputs[2]
                attention_mask = inputs[3]
                rotary_pos_emb = (inputs[4], inputs[5])
                relative_position_bias = inputs[6]
                headscale_tensor = inputs[7]
            else:
                raise ValueError('unexpected number of inputs')
            output_ = self.core_attention(
                query_layer,
                key_layer,
                value_layer,
                attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                relative_position_bias=relative_position_bias,
                headscale_tensor=headscale_tensor,
            )
            return output_

        if rotary_pos_emb is None:
            rot_tuple = (rotary_pos_emb,)
        else:
            rot_tuple = (rotary_pos_emb[0], rotary_pos_emb[1])

        hidden_states = tensor_parallel.checkpoint(
            custom_forward,
            False,
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            *rot_tuple,
            relative_position_bias,
            headscale_tensor,
        )

        return hidden_states

    def _allocate_memory(self, inference_max_sequence_len, batch_size, dtype):
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
            dtype=dtype,
            device=torch.cuda.current_device(),
        )

    def _transpose_last_dim(self, mixed_layer, num_splits, num_splits_first):
        input_shape = mixed_layer.size()
        if num_splits_first:
            """[s, b, num_splits * np * hn]
            -->(view) [s, b, num_splits, np, hn]
            -->(tranpose) [s, b, np, num_splits, hn]
            -->(view) [s, b, np * num_splits * hn] """

            intermediate_shape = input_shape[:-1] + (
                num_splits,
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )

            mixed_layer = mixed_layer.view(*intermediate_shape)
            mixed_layer = mixed_layer.transpose(-2, -3).contiguous()
        else:
            """[s, b, np * hn * num_splits]
            -->(view) [s, b, np, hn, num_splits]
            -->(tranpose) [s, b, np, num_splits, hn]
            -->(view) [s, b, np * num_splits * hn] """

            intermediate_shape = input_shape[:-1] + (
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
                num_splits,
            )

            mixed_layer = mixed_layer.view(*intermediate_shape)
            mixed_layer = mixed_layer.transpose(-1, -2).contiguous()
        mixed_layer = mixed_layer.view(*input_shape)

        return mixed_layer

    def forward(
        self,
        hidden_states,
        attention_mask,
        layer_past=None,
        get_key_value=False,
        encoder_output=None,
        set_inference_key_value_memory=False,
        inference_max_sequence_len=None,
        rotary_pos_emb=None,  # rotary positional embedding
        relative_position_bias=None,
        checkpoint_core_attention=False,
    ):
        # hidden_states: [sq, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        if set_inference_key_value_memory:
            assert inference_max_sequence_len and inference_max_sequence_len > 0
            self.inference_key_memory = self._allocate_memory(
                inference_max_sequence_len, hidden_states.size(1), hidden_states.dtype
            )
            self.inference_value_memory = self._allocate_memory(
                inference_max_sequence_len, hidden_states.size(1), hidden_states.dtype
            )
            self.inference_current_sequence_len = 0

        # Some consistency check.
        if inference_max_sequence_len:
            assert self.inference_current_sequence_len < self.inference_key_memory.size(0)
            assert inference_max_sequence_len == self.inference_key_memory.size(0)
        # This is added for safety. In case inference_max_sequence_len
        # is not provided, make sure there is no potential memory left
        # from previous inference.
        if not inference_max_sequence_len:
            self.inference_key_memory = None
            self.inference_value_memory = None

        # =====================
        # Query, Key, and Value
        # =====================

        side_bias_idx = None

        if self.global_token_mode == "transient":
            avg_hidden_states = _pad_to_multiple(hidden_states, self.global_tokens_spacing, 0)
            avg_hidden_states = avg_hidden_states.reshape(
                self.global_tokens_spacing, -1, hidden_states.shape[-2], hidden_states.shape[-1]
            )
            avg_hidden_states = avg_hidden_states.mean(dim=0)
            avg_hidden_states = self.transient_norm(avg_hidden_states)

            side_bias_idx = torch.arange(
                self.global_tokens_spacing // 2,
                self.global_tokens_spacing // 2 + hidden_states.shape[0],
                self.global_tokens_spacing,
                device=hidden_states.device,
            )
            side_bias_idx = side_bias_idx[None, :].expand(hidden_states.shape[1], -1)

            hidden_states = torch.cat((avg_hidden_states, hidden_states), dim=0)
            total_transient_tokens = avg_hidden_states.shape[0]
        else:
            total_transient_tokens = 0

            if self.global_token_mode == "equal_spacing":
                side_bias_idx = torch.arange(
                    0, hidden_states.shape[0], self.global_tokens_spacing, device=hidden_states.device
                )
            elif self.global_token_mode == "start_and_equal_spacing":
                side_bias_idx = torch.arange(self.global_tokens, device=hidden_states.device)
                side_bias_idx = torch.cat(
                    (
                        side_bias_idx,
                        torch.arange(
                            self.global_tokens,
                            hidden_states.shape[0],
                            self.global_tokens_spacing,
                            device=hidden_states.device,
                        ),
                    ),
                    dim=0,
                )
            elif self.global_token_mode == "random_spacing":
                random_spacing = torch.randint(self.global_tokens_spacing, self.global_tokens_spacing * 2, (1,)).item()
                side_bias_idx = torch.arange(0, hidden_states.shape[0], random_spacing, device=hidden_states.device)
            elif self.global_token_mode == "equal_spacing_random_offset":
                side_bias_idx = torch.arange(
                    0, hidden_states.shape[0], self.global_tokens_spacing, device=hidden_states.device
                )
                random_offset = torch.randint(
                    0, self.global_tokens_spacing, side_bias_idx.shape, device=hidden_states.device
                )
                side_bias_idx += random_offset
                side_bias_idx = torch.clamp(side_bias_idx, max=hidden_states.shape[0] - 1)
            elif self.global_token_mode == "random":
                side_bias_idx = torch.randperm(hidden_states.shape[0], device=hidden_states.device)[
                    : self.global_tokens
                ]
            else:
                side_bias_idx = None

            if side_bias_idx is not None:
                side_bias_idx = side_bias_idx[None, :].expand(hidden_states.shape[1], -1)

        global_query_layer = global_key_layer = global_value_layer = None
        if self.attention_type == AttnType.self_attn:
            # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
            mixed_x_layer, _ = self.query_key_value(hidden_states)

            # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                3
                * self.hidden_size_per_attention_head
                * (2 if self.use_long_attention and self.global_tokens > 0 and self.global_attn_separate else 1),
            )
            if self.megatron_legacy:
                mixed_x_layer = self._transpose_last_dim(mixed_x_layer, 3, True)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            if self.use_long_attention and self.global_tokens > 0 and self.global_attn_separate:
                # [sq, b, np, 6 * hn] --> 3 [sq, b, np, hn]
                (
                    query_layer,
                    key_layer,
                    value_layer,
                    global_query_layer,
                    global_key_layer,
                    global_value_layer,
                ) = tensor_parallel.split_tensor_along_last_dim(mixed_x_layer, 6)
            else:
                # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
                (query_layer, key_layer, value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_x_layer, 3)
        else:
            # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
            mixed_kv_layer, _ = self.key_value(encoder_output)

            # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
            if not self.multi_query_attention:
                new_tensor_shape = mixed_kv_layer.size()[:-1] + (
                    self.num_attention_heads_per_partition,
                    2 * self.hidden_size_per_attention_head,
                )
            else:
                new_tensor_shape = mixed_kv_layer.size()[:-1] + (1, 2 * self.hidden_size_per_attention_head,)
            if self.megatron_legacy:
                mixed_kv_layer = self._transpose_last_dim(mixed_kv_layer, 2, True)
            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
            (key_layer, value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_kv_layer, 2)

            # Attention head [sq, b, h] --> [sq, b, hp]
            query_layer, _ = self.query(hidden_states)
            # [sq, b, hp] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
            query_layer = query_layer.view(*new_tensor_shape)

        if self.is_adapter_available():
            key_infused_adapter = self.get_adapter_module(AdapterName.KEY_INFUSED)
            value_infused_adapter = self.get_adapter_module(AdapterName.VALUE_INFUSED)
            if key_infused_adapter:
                assert value_infused_adapter is not None, "Expected value_infused_adapter not found!"
                kls = key_layer.shape
                key_layer = key_infused_adapter(key_layer.reshape(kls[0], kls[1], -1)).reshape(kls)
            if value_infused_adapter:
                assert key_infused_adapter is not None, "Expected key_infused_adapter not found!"
                vls = value_layer.shape
                value_layer = value_infused_adapter(value_layer.reshape(vls[0], vls[1], -1)).reshape(vls)

        # ===================================================
        # Adjust key, value, and attention mask for inference
        # ===================================================

        # duplicate the pos_emb for self attention
        if rotary_pos_emb is not None:
            rotary_pos_emb = rotary_pos_emb if isinstance(rotary_pos_emb, tuple) else ((rotary_pos_emb,) * 2)

        if inference_max_sequence_len:
            # Adjust the range variables.
            start = self.inference_current_sequence_len
            self.inference_current_sequence_len += key_layer.size(0)
            end = self.inference_current_sequence_len
            # Copy key and values.
            self.inference_key_memory[start:end, ...] = key_layer
            self.inference_value_memory[start:end, ...] = value_layer
            key_layer = self.inference_key_memory[:end, ...]
            value_layer = self.inference_value_memory[:end, ...]
            # Adjust attention mask
            attention_mask = attention_mask[..., start:end, :end]
            # adjust the key rotary positional embedding
            if rotary_pos_emb is not None:
                q_pos_emb, k_pos_emb = rotary_pos_emb
                if not set_inference_key_value_memory:
                    # In inference, we compute one token at a time.
                    # Select the correct positional embedding.
                    q_pos_emb = q_pos_emb[end - 1 : end]
                k_pos_emb = k_pos_emb[:end, :, :, :]
                rotary_pos_emb = (q_pos_emb, k_pos_emb)

        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer), value_layer), dim=0)

        if get_key_value:
            present = (key_layer, value_layer)

        if self.position_embedding_type.lower() == 'xpos':
            b = query_layer.shape[1]
            query_layer = rearrange(query_layer, 'sq b np hn -> (b np) sq hn')
            key_layer = rearrange(key_layer, 'sk b np hn -> (b np) sk hn')
            query_layer = self.xpos(query_layer, offset=0 if inference_max_sequence_len is None else end - 1, downscale=False)
            key_layer = self.xpos(key_layer, offset=0, downscale=True)
            query_layer = rearrange(query_layer, '(b np) sq hn -> sq b np hn', b=b)
            key_layer = rearrange(key_layer, '(b np) sk hn -> sk b np hn', b=b)        
    
            
        if checkpoint_core_attention:
            context_layer = self._checkpointed_attention_forward(
                query_layer,
                key_layer,
                value_layer,
                attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                relative_position_bias=relative_position_bias,
                headscale_tensor=self.head_scale_tensor if self.headscale else None,
            )
        else:
            context_layer = self.core_attention(
                query_layer,
                key_layer,
                value_layer,
                attention_mask,
                global_query_layer=global_query_layer,
                global_key_layer=global_key_layer,
                global_value_layer=global_value_layer,
                layer_past=layer_past,
                get_key_value=get_key_value,
                rotary_pos_emb=rotary_pos_emb,
                relative_position_bias=relative_position_bias,
                headscale_tensor=self.head_scale_tensor if self.headscale else None,
                total_transient_tokens=total_transient_tokens,
                side_bias_idx=side_bias_idx,
            )

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)
        if get_key_value:
            output = [output, present]

        return output, bias


class ParallelChunkedCrossAttention(MegatronModule):
    """Parallel chunked cross-attention layer class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        init_method,
        output_layer_init_method,
        layer_number,
        num_attention_heads,
        hidden_size,
        precision=16,
        apply_query_key_layer_scaling=True,
        kv_channels=None,
        use_cpu_initialization=False,
        masked_softmax_fusion=True,
        attention_dropout=0.1,
        megatron_legacy=False,
        chunk_size=64,  # each chunk, how many tokens
        bias=True,
        headscale=False,
        gradient_accumulation_fusion=False,
        normalize_attention_scores=True,
    ):
        super(ParallelChunkedCrossAttention, self).__init__()
        self.cross_attention = ParallelAttention(
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            layer_number=layer_number,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            attention_type=AttnType.cross_attn,
            attn_mask_type=AttnMaskType.padding,
            precision=precision,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            kv_channels=kv_channels,
            use_cpu_initialization=use_cpu_initialization,
            masked_softmax_fusion=masked_softmax_fusion,
            attention_dropout=attention_dropout,
            megatron_legacy=megatron_legacy,
            bias=bias,
            headscale=headscale,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
            normalize_attention_scores=normalize_attention_scores,
        )
        self.chunk_size = chunk_size

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_output=None,
        set_inference_key_value_memory=False,
        inference_max_sequence_len=None,
        rotary_pos_emb=None,
        checkpoint_core_attention=False,
    ):
        if checkpoint_core_attention:
            raise ValueError(
                'checkpoint_core_attention during forward not implemented yet for ParallelChunkedCrossAttention'
            )

        # hidden_states is assumed to have dimension [token length, batch, dimension]
        # derive variables
        # encoder_output here is the retrieved context
        context = encoder_output
        # context is assumed to have dimension [num_chunks, num_neighbors, context_token_len, batch, dimension]
        chunk_size = self.chunk_size
        b, n, dim = (
            hidden_states.shape[1],
            hidden_states.shape[0],
            hidden_states.shape[2],
        )
        default_bias = self.cross_attention.dense.bias
        if set_inference_key_value_memory:
            seq_index = (n // chunk_size) * chunk_size
            self.current_len = n
        elif inference_max_sequence_len is not None:
            # only handles single token increment
            assert n == 1
            self.current_len += n
            chunk_id = self.current_len // chunk_size
            if chunk_id <= 0:
                # if sequence length less than chunk size, do an early return
                return torch.zeros_like(hidden_states), default_bias
            causal_padding = chunk_size - 1
            # pad it as a full chunk, put it at the end of the chunk position
            hidden_states = F.pad(hidden_states, (0, 0, 0, 0, causal_padding, 0), value=0.0)
            # only use the relevant context
            context = context[chunk_id - 1 : chunk_id, :, :, :, :]
            attention_mask = rearrange(attention_mask, '(b k) 1 q v -> b k 1 q v', b=b)
            # select the relevant chunk attn mask
            attention_mask = attention_mask[:, chunk_id - 1]
            seq_index = chunk_size
        else:
            # this is normal forward without inference
            seq_index = (n // chunk_size) * chunk_size

        # if sequence length less than chunk size, do an early return
        if n < self.chunk_size and set_inference_key_value_memory and inference_max_sequence_len is not None:
            return torch.zeros_like(hidden_states), default_bias

        num_chunks, num_retrieved = (
            context.shape[-5],
            context.shape[-4],
        )

        # causal padding
        causal_padding = chunk_size - 1

        x = F.pad(hidden_states, (0, 0, 0, 0, -causal_padding, causal_padding), value=0.0)

        # remove sequence which is ahead of the neighbors retrieved (during inference)

        # seq_index = (n // chunk_size) * chunk_size
        x, x_remainder = x[:seq_index], x[seq_index:]

        seq_remain_len = x_remainder.shape[0]

        # take care of rotary positional embedding
        # make sure queries positions are properly shifted to the future

        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            # currently implementation is broken
            # q need to extend to causal_padding, and just do
            # q_pos_emb = F.pad(q_pos_emb, (0, 0, -causal_padding, 0), value = 0.)
            if inference_max_sequence_len is not None and not set_inference_key_value_memory:
                token_pos = (self.current_len - 1) % chunk_size
                q_pos_emb = F.pad(
                    q_pos_emb, (0, 0, 0, 0, 0, 0, -causal_padding - token_pos, -causal_padding + token_pos), value=0.0
                )
            else:
                q_pos_emb = F.pad(q_pos_emb, (0, 0, 0, 0, 0, 0, -causal_padding, 0), value=0.0)

            k_pos_emb = repeat(k_pos_emb, 'n b h d -> (r n) b h d', r=num_retrieved)
            rotary_pos_emb = (q_pos_emb, k_pos_emb)

        # make sure number context chunks is enough
        assert x.shape[0] // chunk_size == num_chunks

        # reshape so we have chunk to chunk attention, without breaking causality
        x = rearrange(x, '(k n) b d -> n (b k) d', k=num_chunks)
        context = rearrange(context, 'k r n b d -> (r n) (b k) d')
        # cross attention
        out, bias = self.cross_attention(x, attention_mask, encoder_output=context, rotary_pos_emb=rotary_pos_emb)

        # reshape back to original sequence

        out = rearrange(out, 'n (b k) d -> (k n) b d', b=b)

        # pad back to original, with 0s at the beginning (which will be added to the residual and be fine)

        out = F.pad(out, (0, 0, 0, 0, causal_padding, -causal_padding + seq_remain_len), value=0.0)
        if not set_inference_key_value_memory and inference_max_sequence_len is not None:
            out = out[-1:]
        return out, bias


class CoreAttention(MegatronModule):
    """ Region where selective activation recomputation is applied.
        See Figure 3. in Reducing Activation Recomputation in Large Transformer Models
        https://arxiv.org/pdf/2205.05198.pdf for more details.

    """

    def __init__(
        self,
        layer_number,
        num_attention_heads,
        hidden_size,
        attention_type=AttnType.self_attn,
        attn_mask_type=AttnMaskType.padding,
        precision=16,
        apply_query_key_layer_scaling=True,
        kv_channels=None,
        masked_softmax_fusion=True,
        attention_dropout=0.1,
        sequence_parallel=False,
        normalize_attention_scores=True,
        multi_query_attention=False,
        position_embedding_type='learned_absolute',
        use_long_attention=False,
        local_context=128,
        global_tokens=1024,
        global_tokens_spacing=16,
        global_attn_separate=True,
        transient_global_tokens=False,
        use_flash_attention=False,
    ):

        super(CoreAttention, self).__init__()

        self.precision = precision
        self.fp16 = precision == 16
        self.bf16 = precision == 'bf16'
        self.multi_query_attention = multi_query_attention
        self.position_embedding_type = position_embedding_type

        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = False
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.sequence_parallel = sequence_parallel
        # If True, will scale attention scores by 1 / sqrt(hidden_size_per_attention_head).
        # This arg is been provided mostly to support weight conversion of Huggingface models. (ex: T5v1.1)
        self.normalize_attention_scores = normalize_attention_scores

        if kv_channels is None:
            assert (
                hidden_size % num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = hidden_size // num_attention_heads

        projection_size = kv_channels * num_attention_heads

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = safe_divide(projection_size, world_size)
        self.hidden_size_per_attention_head = safe_divide(projection_size, num_attention_heads)
        self.num_attention_heads_per_partition = safe_divide(num_attention_heads, world_size)
        self.num_attention_heads_partition_offset = (
            self.num_attention_heads_per_partition * parallel_state.get_tensor_model_parallel_rank()
        )

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = MatchedScaleMaskSoftmax(
            self.fp16,
            self.bf16,
            self.attn_mask_type,
            masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff,
        )

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout_p = attention_dropout
        self.attention_dropout = torch.nn.Dropout(attention_dropout)

        self.use_long_attention = use_long_attention
        self.local_context = local_context
        self.global_tokens = global_tokens
        self.global_tokens_spacing = global_tokens_spacing
        self.global_attn_separate = global_attn_separate

        self.transient_global_tokens = transient_global_tokens
        self.use_flash_attention = use_flash_attention

    def forward(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        global_query_layer=None,
        global_key_layer=None,
        global_value_layer=None,
        layer_past=None,
        get_key_value=False,
        rotary_pos_emb=None,
        relative_position_bias=None,
        headscale_tensor=None,
        total_transient_tokens=0,
        side_bias_idx=None,
    ):
        b, np, sq, sk, hn = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0) - total_transient_tokens,
            key_layer.size(0) - total_transient_tokens,
            query_layer.size(3),
        )

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================
        if get_key_value:
            with torch.no_grad():
                if layer_past is not None:
                    attention_mask = attention_mask[..., sq - 1, :sk].unsqueeze(2)
                else:
                    attention_mask = attention_mask[..., :sq, :sk]

        # ==================================================
        # Update attention bias.
        # relative_position_bias: [1, np, sq, sk]
        # relative_position_bias w/ long-attention:  [1, np, nb, sq, sk]
        # attention_bias: [b, np, sq, sk]
        # ==================================================
        attention_bias = None
        if self.position_embedding_type.lower() == 'sandwich':
            attention_bias = sandwich_pos_bias(
                sq, sk, self.hidden_size_per_attention_head, np, torch.cuda.current_device()
            )

        if relative_position_bias is not None:
            attention_bias = relative_position_bias[
                :,
                self.num_attention_heads_partition_offset : self.num_attention_heads_partition_offset
                + self.num_attention_heads_per_partition,
            ]
            attention_bias = attention_bias[..., -sq:, -sk:]
            if attention_bias.shape[0] == 1:
                attention_bias = attention_bias.expand(b, *([-1] * (attention_bias.dim() - 1)))

        # ==================================================
        # Update query_layer, key_layer, value_layer.
        # query_layer: [sq, b, np, hn]
        # key_layer: [sk, b, np, hn]
        # ==================================================
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            query_layer = apply_rotary_pos_emb(query_layer, q_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, k_pos_emb)
            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)


        # ==================================================
        # Long Attention
        # ==================================================
        if self.use_long_attention:
            (query_layer, key_layer, value_layer, attention_mask, attention_bias,) = self.long_attention(
                query_layer,
                key_layer,
                value_layer,
                attention_mask,
                attention_bias,
                total_transient_tokens,
                side_bias_idx,
            )

        # ==================================================
        # Rearrange query_layer, key_layer, value_layer.
        # ==================================================
        query_layer = rearrange(query_layer, 'sq b np hn -> b sq np hn')
        key_layer = rearrange(key_layer, 'sk b np hn -> b sk np hn')
        value_layer = rearrange(value_layer, 'sk b np hn -> b sk np hn')

        # ==================================================
        # Get context_layer [b, np, sq, hn]
        # ==================================================
        if self.use_flash_attention:
            (query_layer, key_layer, value_layer, attention_mask, attention_bias) = _cast_if_autocast_enabled(
                query_layer, key_layer, value_layer, attention_mask, attention_bias,
            )

            if attention_bias is not None:
                context_layer = self.flash_attention_triton(
                    query_layer, key_layer, value_layer, attention_mask, attention_bias
                )
            else:
                context_layer = self.flash_attention(query_layer, key_layer, value_layer, attention_mask,)

        else:
            context_layer = self.torch_attention(query_layer, key_layer, value_layer, attention_mask, attention_bias, self.multi_query_attention)

        if self.use_long_attention:
            assert sq == sk, 'Long Attention can only be used for self-attention.'
            nb = (sq + self.local_context - 1) // self.local_context
            context_layer = rearrange(context_layer, '(b nb) np sq hn -> b np (nb sq) hn', nb=nb)
            context_layer = context_layer[:, :, :sq]

        if headscale_tensor is not None:
            context_layer = context_layer * headscale_tensor

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

    def get_local_pos_bias(self, relative_position_bias, num_blocks, block_length):
        memory_position = torch.arange(3 * block_length, dtype=torch.long, device=torch.cuda.current_device())
        context_position = memory_position[block_length:-block_length]

        relative_position = memory_position[None, :] - context_position[:, None]

        bias = relative_position_bias.get_bias(relative_position[None, :])[:, :, None]

        return bias

    def get_side_pos_bias(self, relative_position_bias, num_blocks, block_length, side_bias_idx):
        context_position = torch.arange(num_blocks * block_length, device=side_bias_idx.device)

        # bs, time, side
        relative_position = side_bias_idx[:, None, :] - context_position[None, :, None]

        bias = relative_position_bias.get_bias(relative_position)
        bias = bias.reshape(bias.shape[0], bias.shape[1], num_blocks, block_length, -1)

        return bias

    def long_attention(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        attention_bias,
        total_transient_tokens,
        side_bias_idx,
    ):
        # ==================================================
        # Update query_layer, key_layer, value_layer
        # ==================================================
        if side_bias_idx is not None:
            if self.transient_global_tokens:
                side_k = key_layer[:total_transient_tokens]
                side_v = value_layer[:total_transient_tokens]

                side_k = side_k.permute(1, 2, 0, 3)
                side_v = side_v.permute(1, 2, 0, 3)

                query_layer = query_layer[total_transient_tokens:]
                key_layer = key_layer[total_transient_tokens:]
                value_layer = value_layer[total_transient_tokens:]
            else:
                side_k = torch.gather(
                    key_layer.transpose(0, 1),
                    dim=1,
                    index=side_bias_idx[..., None, None].expand(-1, -1, key_layer.shape[-2], key_layer.shape[-1]),
                )
                side_v = torch.gather(
                    value_layer.transpose(0, 1),
                    dim=1,
                    index=side_bias_idx[..., None, None].expand(-1, -1, value_layer.shape[-2], value_layer.shape[-1]),
                )

                side_k = side_k.transpose(1, 2)
                side_v = side_v.transpose(1, 2)

        # [sq/k, b, np, hn] -> [b, hn, sq/k, np]
        query_layer = query_layer.permute(1, 2, 0, 3)
        key_layer = key_layer.permute(1, 2, 0, 3)
        value_layer = value_layer.permute(1, 2, 0, 3)

        # Split into blocks -> (batch_size, n_heads, num_blocks, block_len, dim_per_head)
        query_layer = _split_into_blocks(query_layer, self.local_context, dim=2)
        key_layer = _split_into_blocks(key_layer, self.local_context, dim=2)
        value_layer = _split_into_blocks(value_layer, self.local_context, dim=2)

        # Concatenate 3 blocks for keys and values -> (batch_size, n_heads, num_blocks, 3 * block_len, dim_per_head)
        key_layer = _concatenate_3_blocks(key_layer, block_dim=2, sequence_dim=3)
        value_layer = _concatenate_3_blocks(value_layer, block_dim=2, sequence_dim=3)

        if side_bias_idx is not None:
            # Tile side inputs across local key/value blocks
            # New shape: (batch_size, n_heads, num_blocks, global_seq_len, dim_per_head)
            reps = [1] * (side_k.ndim + 1)
            reps[2] = key_layer.shape[2]
            side_k = side_k.unsqueeze(2).repeat(reps)
            side_v = side_v.unsqueeze(2).repeat(reps)

            # Concatenate "local" and "side"/"global" key/value states
            # New shape: (batch_size, n_heads, num_blocks, 3 * block_len + global_seq_len, dim_per_head)
            key_layer = torch.cat([key_layer, side_k], dim=3)
            value_layer = torch.cat([value_layer, side_v], dim=3)

        # ==================================================
        # Update attention_mask, attention_bias
        # ==================================================
        # [TODO] unified attention_mask dimension
        # attention_mask: [b, 1, sk]
        bs, np, nb, sk = key_layer.shape[:4]
        sq = query_layer.shape[3]

        local_attention_mask = _get_local_attention_mask(
            ~attention_mask.squeeze(1), self.local_context, attention_mask.device
        ).expand(-1, np, -1, -1, -1)

        # Default attend to all tokens including global tokens
        attention_mask = torch.ones(
            bs, np, nb, sq, sk, dtype=local_attention_mask.dtype, device=torch.cuda.current_device(),
        )
        attention_mask[..., : local_attention_mask.shape[-1]] = local_attention_mask

        if attention_bias is None:
            attention_bias = torch.where(attention_mask, 0.0, -1e10)
        else:
            attention_bias = attention_bias + torch.where(attention_mask, 0.0, -1e10)

        query_layer = rearrange(query_layer, "bs np nb sq hn -> sq (bs nb) np hn")
        key_layer = rearrange(key_layer, "bs np nb sk hn -> sk (bs nb) np hn")
        value_layer = rearrange(value_layer, "bs np nb sv hn -> sv (bs nb) np hn")
        attention_mask = None
        attention_bias = rearrange(attention_bias, 'b np nb sq sk -> (b nb) np sq sk')

        return query_layer, key_layer, value_layer, attention_mask, attention_bias

    def torch_attention(self, query_layer, key_layer, value_layer, attention_mask, attention_bias, multi_query):

        np = query_layer.shape[2]

        if multi_query:
            query_layer = rearrange(query_layer, 'b sq np hn -> b (np sq) hn')
            key_layer = rearrange(key_layer, 'b sk 1 hn -> b hn sk')
            value_layer = rearrange(value_layer, 'b sk np hn -> (b np) sk hn')
        else:
            query_layer = rearrange(query_layer, 'b sq np hn -> (b np) sq hn')
            key_layer = rearrange(key_layer, 'b sk np hn -> (b np) hn sk')
            value_layer = rearrange(value_layer, 'b sk np hn -> (b np) sk hn')

        matmul_input_buffer = torch.empty(
            query_layer.shape[0],
            query_layer.shape[1],
            key_layer.shape[2],
            dtype=query_layer.dtype,
            device=torch.cuda.current_device(),
        )

        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_layer,
            key_layer,
            beta=0.0,
            alpha=(1.0 / self.norm_factor) if self.normalize_attention_scores else 1.0,
        )

        attention_scores = rearrange(matmul_result, '(b np) sq sk -> b np sq sk', np=np)

        if attention_bias is not None:
            attention_scores += attention_bias

        if attention_mask is not None:
            attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)
        else:
            attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        if not self.sequence_parallel:
            with tensor_parallel.random.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        attention_probs = rearrange(attention_probs, 'b np sq sk -> (b np) sq sk')
        context_layer = torch.bmm(attention_probs, value_layer)
        context_layer = rearrange(context_layer, '(b np) sq hn -> b np sq hn', np=np)

        return context_layer

    def flash_attention(self, query_layer, key_layer, value_layer, attention_mask):
        batch_size, seqlen, nheads, _ = query_layer.shape

        # [b, 1, sq, sk] -> [b, sq] / [b, sk]
        # True: not attend / False: attend -> True: attend / False: not attend
        if attention_mask is not None:
            if len(attention_mask.size()) == 4:
                attention_mask_q = torch.any(torch.eq(attention_mask, False), dim=3).squeeze(1)
                attention_mask_kv = torch.any(torch.eq(attention_mask, False), dim=2).squeeze(1)
            elif len(attention_mask.size()) == 3:
                attention_mask_q = attention_mask.squeeze(1)
                attention_mask_kv = attention_mask.squeeze(1)
        else:
            attention_mask_q = torch.ones(
                batch_size, query_layer.shape[1], dtype=torch.bool, device=torch.cuda.current_device()
            )
            attention_mask_kv = torch.ones(
                batch_size, key_layer.shape[1], dtype=torch.bool, device=torch.cuda.current_device()
            )

        q, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(query_layer, attention_mask_q)
        k, _, cu_seqlens_k, max_seqlen_k = unpad_input(key_layer, attention_mask_kv)
        v, _, _, _ = unpad_input(value_layer, attention_mask_kv)

        context_layer = flash_attn_unpadded_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=self.attention_dropout_p if self.training else 0.0,
            causal=self.attn_mask_type == AttnMaskType.causal,
        )

        # [b, sq, np, hn]
        context_layer = pad_input(context_layer, indices_q, batch_size, seqlen)

        # [b, sq, np, hn] -> [b, np, sq, hn]
        context_layer = context_layer.permute(0, 2, 1, 3)
        return context_layer

    def flash_attention_triton(self, query_layer, key_layer, value_layer, attention_mask, attention_bias):
        if self.attention_dropout_p > 0.0:
            raise NotImplementedError(f'attention_dropout not implemented for flash_attention with attention bias')

        # [b, 1, sq, sk] -> [b, 1, 1, sk]
        if attention_mask is not None:
            attention_mask_kv = torch.any(torch.eq(attention_mask, False), dim=2).unsqueeze(2)
            attention_bias = attention_bias.masked_fill(~attention_mask_kv, torch.finfo(query_layer.dtype).min)

        context_layer = flash_attn_func(
            query_layer, key_layer, value_layer, attention_bias, self.attn_mask_type == AttnMaskType.causal
        )

        # [b, sq, np, hn] -> [b, np, sq, hn]
        context_layer = context_layer.permute(0, 2, 1, 3)
        return context_layer

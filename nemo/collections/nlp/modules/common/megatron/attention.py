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
from einops import rearrange, repeat

from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import AdapterName, InfusedAdapterConfig
from nemo.collections.nlp.modules.common.megatron.fused_softmax import MatchedScaleMaskSoftmax
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.rotary_pos_embedding import apply_rotary_pos_emb
from nemo.collections.nlp.modules.common.megatron.xpos_relative_position import XPOS
from nemo.collections.nlp.modules.common.megatron.sandwich_relative_position import sandwich_pos_bias
from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults, attention_mask_func
from nemo.core import adapter_mixins

import math
from functools import lru_cache
from typing import List, Tuple


from nemo.utils import avoid_float16_autocast_context

try:
    from apex.transformer import parallel_state, tensor_parallel
    from apex.transformer.enums import AttnMaskType, AttnType
    from apex.transformer.utils import divide as safe_divide

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

    # fake missing classes with None attributes
    ModelType = AttnMaskType = AttnType = LayerType = ApexGuardDefaults()

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
    ):
        super(ParallelAttention, self).__init__()

        self.use_long_attention = use_long_attention
        self.local_context = local_context
        self.global_tokens = global_tokens
        self.global_tokens_spacing = 16
        self.global_attn_separate = global_attn_separate

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

            self.key_value = tensor_parallel.ColumnParallelLinear(
                hidden_size,
                2 * projection_size,
                gather_output=False,
                init_method=init_method,
                bias=bias,
                sequence_parallel_enabled=sequence_parallel,
                no_async_tensor_model_parallel_allreduce=no_async_tensor_model_parallel_allreduce,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
            )

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
            global_attn_separate=global_attn_separate,
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
            new_tensor_shape = mixed_kv_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                2 * self.hidden_size_per_attention_head,
            )
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
        self.attention_dropout = torch.nn.Dropout(attention_dropout)

        if position_embedding_type.lower() == 'xpos':
            self.xpos = XPOS(hidden_size / num_attention_heads)

        self.use_long_attention = use_long_attention
        self.local_context = local_context
        self.global_tokens = global_tokens
        self.global_tokens_spacing = global_tokens_spacing
        self.global_attn_separate = global_attn_separate

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
    ):

        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.size(1), query_layer.size(2), query_layer.size(0), key_layer.size(0))

        # TODO: figure out how to do this
        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            query_layer = apply_rotary_pos_emb(query_layer, q_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, k_pos_emb)
            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)

        if self.multi_query_attention:
            # [sq, b, np, hn] -> [b, np * sq, hn]
            query_layer = query_layer.permute([1, 2, 0, 3]).reshape(
                output_size[0], output_size[1] * output_size[2], -1
            )

            # [sk, b, 1, hn] -> [b, hn, sk]
            key_layer = key_layer.squeeze(2).permute(1, 2, 0)

            # preallocting input tensor: [b * np, sq, sk]
            matmul_input_buffer = torch.empty(
                output_size[0] * output_size[1],
                output_size[2],
                output_size[3],
                dtype=query_layer.dtype,
                device=torch.cuda.current_device(),
            )

            # Raw attention scores. [b * np, sq, sk]
            matmul_result = torch.baddbmm(
                matmul_input_buffer,
                query_layer,  # [b * np, sq, hn]
                key_layer,  # [b * np, hn, sk]
                beta=0.0,
                alpha=(1.0 / self.norm_factor),
            )
        else:
            if self.position_embedding_type.lower() == 'xpos':
                sq, bs, hn, np = query_layer.shape
                query_layer = rearrange(query_layer, 's b h d -> (b h) s d')
                key_layer = rearrange(key_layer, 's b h d -> (b h) s d')
                key_layer = self.xpos(key_layer, offset=0, downscale=True)
                query_layer = self.xpos(query_layer, offset=sq - 1, downscale=False)
                # permute back to the expected shape below
                key_layer = key_layer.permute(1, 0, 2)
                query_layer = query_layer.permute(1, 0, 2)

                if self.use_long_attention and self.global_attn_separate:
                    global_query_layer = rearrange(global_query_layer, 's b h d -> (b h) s d')
                    global_key_layer = rearrange(global_key_layer, 's b h d -> (b h) s d')
                    global_key_layer = self.xpos(global_key_layer, offset=0, downscale=True)
                    global_query_layer = self.xpos(global_query_layer, offset=0, downscale=False)
                    # permute back to the expected shape below
                    global_key_layer = global_key_layer.permute(1, 0, 2)
                    global_query_layer = global_query_layer.permute(1, 0, 2)

            if not self.use_long_attention:
                # [sq, b, np, hn] -> [sq, b * np, hn]
                query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
                # [sk, b, np, hn] -> [sk, b * np, hn]
                key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

                # preallocting input tensor: [b * np, sq, sk]
                matmul_input_buffer = torch.empty(
                    output_size[0] * output_size[1],
                    output_size[2],
                    output_size[3],
                    dtype=query_layer.dtype,
                    device=torch.cuda.current_device(),
                )

                # Raw attention scores. [b * np, sq, sk]
                matmul_result = torch.baddbmm(
                    matmul_input_buffer,
                    query_layer.transpose(0, 1),  # [b * np, sq, hn]
                    key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                    beta=0.0,
                    alpha=(1.0 / self.norm_factor) if self.normalize_attention_scores else 1.0,
                )

            else:

                if len(attention_mask.shape) > 3:
                    attention_mask = attention_mask.sum(-1) == 0
                    # attention_mask = attention_mask.transpose(-2, -1)
                # [sq/k, b, np, hn]
                query_layer = query_layer.view(output_size[2], output_size[0], output_size[1], -1)
                key_layer = key_layer.view(output_size[3], output_size[0], output_size[1], -1)
                value_layer = value_layer.view(output_size[3], output_size[0], output_size[1], -1)

                # [b, hn, sq/k, np]
                query_layer = query_layer.permute(1, 2, 0, 3)
                key_layer = key_layer.permute(1, 2, 0, 3)
                value_layer = value_layer.permute(1, 2, 0, 3)

                T_q = query_layer.shape[2]
                T_k = key_layer.shape[2]
                pad_len_q = (2 * self.local_context - T_q % (2 * self.local_context)) % (2 * self.local_context)
                pad_len_k = (2 * self.local_context - T_k % (2 * self.local_context)) % (2 * self.local_context)

                query_layer = F.pad(query_layer, (0, 0, 0, pad_len_q))  # (batch, head, time, size)
                key_layer = F.pad(key_layer, (0, 0, 0, pad_len_k))  # (batch, head, time, size)
                value_layer = F.pad(value_layer, (0, 0, 0, pad_len_k))  # (batch, head, time, size)
                attention_mask = F.pad(attention_mask, (0, pad_len_q), value=True)

                # [b, hn, sq, 2w+1]
                attention_scores = self.sliding_chunks_matmul_qk(
                    query_layer, key_layer, self.local_context, padding_value=0
                )

                attention_mask = attention_mask.unsqueeze(dim=-1)
                float_mask = attention_mask.type_as(attention_scores).masked_fill(attention_mask, -10000.0)
                ones = float_mask.new_ones(size=float_mask.size())  # tensor of ones
                # diagonal mask with zeros everywhere and -inf inplace of padding
                d_mask = self.sliding_chunks_matmul_qk(ones, float_mask, self.local_context, padding_value=0.0)
                # (batch, head, time, 2w + 1)

                d_mask = d_mask < -1
                # attention_scores += d_mask

                # attention_probs = F.softmax(attention_scores, dim=-1)
                attention_probs = self.scale_mask_softmax(attention_scores, d_mask)

                if not self.sequence_parallel:
                    with tensor_parallel.random.get_cuda_rng_tracker().fork():
                        attention_probs = self.attention_dropout(attention_probs)
                else:
                    attention_probs = self.attention_dropout(attention_probs)

                # matmul: [b * np, sq, hn]
                context_layer = self.sliding_chunks_matmul_pv(attention_probs, value_layer, self.local_context)
                context_layer = context_layer.reshape(context_layer.shape[0], context_layer.shape[1], -1)

                if self.global_tokens > 0:

                    # create q, k, v for global attn
                    if self.global_attn_separate:
                        global_q, global_k, global_v = global_query_layer, global_key_layer, global_value_layer
                        global_q = global_q.view(output_size[2], output_size[0], output_size[1], -1)
                        global_k = global_k.view(output_size[3], output_size[0], output_size[1], -1)
                        global_v = global_v.view(output_size[3], output_size[0], output_size[1], -1)
                        global_q = global_q.permute(1, 2, 0, 3)
                        global_k = global_k.permute(1, 2, 0, 3)
                        global_v = global_v.permute(1, 2, 0, 3)
                        global_q = F.pad(global_q, (0, 0, 0, pad_len_q))  # (batch, head, time, size)
                        global_k = F.pad(global_k, (0, 0, 0, pad_len_k))  # (batch, head, time, size)
                        global_v = F.pad(global_v, (0, 0, 0, pad_len_k))  # (batch, head, time, size)
                    else:
                        global_q, global_k, global_v = query_layer, key_layer, value_layer

                    # assign which tokens are global
                    is_index_global_attn = torch.zeros_like(attention_mask.squeeze(3).squeeze(1))
                    is_index_global_attn[
                        :, : self.global_tokens * self.global_tokens_spacing : self.global_tokens_spacing
                    ] = 1.0

                    # compute global attn indices
                    (
                        max_num_global_attn_indices,
                        is_index_global_attn_nonzero,
                        is_local_index_global_attn_nonzero,
                        is_local_index_no_global_attn_nonzero,
                    ) = self._get_global_attn_indices(is_index_global_attn=is_index_global_attn)

                    # calculate global attn probs with global keys
                    # (batch, time, head, max_num_global_attn_indices)
                    global_key_attn = self._compute_global_key_attn(
                        query=global_q.transpose(1, 2),
                        key=global_k.transpose(1, 2),
                        max_num_global_attn_indices=max_num_global_attn_indices,
                        is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                        is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                        is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
                    ).transpose(1, 2)

                    # global_key_attn = torch.softmax(global_key_attn, dim=-1).masked_fill(attention_mask, 0.0)
                    rep_mask = repeat(attention_mask, 'b h t m -> b h t (m m2)', m2=global_key_attn.shape[3])
                    global_key_attn = self.scale_mask_softmax(global_key_attn, rep_mask)
                    if not self.sequence_parallel:
                        with tensor_parallel.random.get_cuda_rng_tracker().fork():
                            global_key_attn = self.attention_dropout(global_key_attn)
                    else:
                        global_key_attn = self.attention_dropout(global_key_attn)

                    # compute outputs for global attention from all tokens to global
                    # (batch, time, head x head_dim)
                    out_all_to_global = self._compute_out_all_to_global(
                        value=global_v,
                        attn_probs=global_key_attn,
                        max_num_global_attn_indices=max_num_global_attn_indices,
                        is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                        is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                    )

                    # compute outputs for global attention from global tokens to all
                    # (batch, max_num_global_attn_indices, head x head_dim)
                    out_global_to_all = self._compute_out_global_to_all(
                        query=global_q,
                        key=global_k,
                        value=global_v,
                        max_num_global_attn_indices=max_num_global_attn_indices,
                        is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                        is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                        is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
                        is_index_masked=attention_mask,
                    )

                    context_layer += out_all_to_global

                    context_layer[is_index_global_attn_nonzero] += out_global_to_all

                context_layer = context_layer[:, : output_size[2]]

                context_layer = context_layer.transpose(0, 1).contiguous()
                # [batch, seq, head, dim]

        # change view to [b, np, sq, sk]
        if not self.use_long_attention:
            attention_scores = matmul_result.view(*output_size)
            if self.position_embedding_type.lower() == 'sandwich':
                b, np, sq, sk = attention_scores.shape
                sandwich_bias = sandwich_pos_bias(
                    sq, sk, self.hidden_size_per_attention_head, np, torch.cuda.current_device()
                )
                attention_scores += sandwich_bias

            if relative_position_bias is not None:
                attention_scores += relative_position_bias[
                    :,
                    self.num_attention_heads_partition_offset : self.num_attention_heads_partition_offset
                    + self.num_attention_heads_per_partition,
                    : attention_scores.size(2),
                    : attention_scores.size(3),
                ]

            # ==================================================
            # Update attention mask for inference. [b, np, sq, sk]
            # ==================================================

            if get_key_value:
                with torch.no_grad():
                    if layer_past is not None:
                        attention_mask = attention_mask[
                            ..., attention_scores.size(3) - 1, : attention_scores.size(3)
                        ].unsqueeze(2)
                    else:
                        attention_mask = attention_mask[..., : attention_scores.size(3), : attention_scores.size(3)]

            # ===========================
            # Attention probs and dropout
            # ===========================

            # attention scores and attention mask [b, np, sq, sk]
            attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.

            if not self.sequence_parallel:
                with tensor_parallel.random.get_cuda_rng_tracker().fork():
                    attention_probs = self.attention_dropout(attention_probs)
            else:
                attention_probs = self.attention_dropout(attention_probs)

            # =========================
            # Context layer. [sq, b, hp]
            # =========================

            # value_layer -> context layer.
            # [sk, b, np, hn] --> [b, np, sq, hn]

            # context layer shape: [b, np, sq, hn]
            output_size = (value_layer.size(1), value_layer.size(2), query_layer.size(0), value_layer.size(3))

            # change view [sk, b * np, hn]
            value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)

            # change view [b * np, sq, sk]
            attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

            # matmul: [b * np, sq, hn]
            context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

            # change view [b, np, sq, hn]
            context_layer = context_layer.view(*output_size)

            if headscale_tensor is not None:
                context_layer = context_layer * headscale_tensor

            # [b, np, sq, hn] --> [sq, b, np, hn]
            context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

            # [sq, b, np, hn] --> [sq, b, hp]
            new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
            context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

    def _get_global_attn_indices(self, is_index_global_attn: torch.Tensor) -> Tuple:
        """
        Compute global attention indices.

        Args:
            is_index_global_attn (torch.Tensor): (batch, time) A boolean tensor indicating if an index is a global attention index.

        Returns:
            max_num_global_attn_indices (int): Maximum number of global attention indices in the batch.
            is_index_global_attn_nonzero (tuple): Indices of global attention (non-zero elements).
            is_local_index_global_attn_nonzero (tuple): Indices of non-padding values within global attention indices.
            is_local_index_no_global_attn_nonzero (tuple): Indices of padding values within global attention indices.
        """
        # Calculate the number of global attention indices in the batch
        num_global_attn_indices = is_index_global_attn.long().sum(dim=1)

        # Find the maximum number of global attention indices in the batch
        max_num_global_attn_indices = num_global_attn_indices.max()

        # Get the indices of global attention (non-zero elements)
        is_index_global_attn_nonzero = is_index_global_attn.nonzero(as_tuple=True)

        # Create a helper tensor to find the local indices of global attention
        is_local_index_global_attn = torch.arange(
            max_num_global_attn_indices, device=is_index_global_attn.device
        ) < num_global_attn_indices.unsqueeze(dim=-1)

        # Find the non-padding values within global attention indices
        is_local_index_global_attn_nonzero = is_local_index_global_attn.nonzero(as_tuple=True)

        # Find the padding values within global attention indices
        is_local_index_no_global_attn_nonzero = (is_local_index_global_attn == 0).nonzero(as_tuple=True)

        return (
            max_num_global_attn_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attn_nonzero,
            is_local_index_no_global_attn_nonzero,
        )

    def _compute_global_key_attn(
        self,
        key: torch.Tensor,
        query: torch.Tensor,
        max_num_global_attn_indices: int,
        is_index_global_attn_nonzero: tuple,
        is_local_index_global_attn_nonzero: tuple,
        is_local_index_no_global_attn_nonzero: tuple,
    ) -> torch.Tensor:
        """
        Compute the attention probabilities using only global key vectors.

        Args:
            key (torch.Tensor): (batch, time, head, head_dim) The key vectors.
            query (torch.Tensor): (batch, time, head, head_dim) The query vectors.
            max_num_global_attn_indices (int): Maximum number of global attention indices in the batch.
            is_index_global_attn_nonzero (tuple): Indices of global attention (non-zero elements).
            is_local_index_global_attn_nonzero (tuple): Non-padding values within global attention indices.
            is_local_index_no_global_attn_nonzero (tuple): Padding values within global attention indices.

        Returns:
            attn_probs_from_global_key (torch.Tensor): (batch, time, head, max_num_global_attn_indices) The computed attention probabilities using only global key vectors.
        """
        batch_size, h, d_k = key.shape[0], key.shape[2], key.shape[3]

        # create only global key vectors
        key_only_global = key.new_zeros(batch_size, max_num_global_attn_indices, h, d_k)

        key_only_global[is_local_index_global_attn_nonzero] = key[is_index_global_attn_nonzero]

        # (batch_size, seq_len, head, max_num_global_attn_indices)
        attn_probs_from_global_key = torch.einsum("blhd,bshd->blhs", (query, key_only_global))

        # need to transpose since ONNX export only supports consecutive indexing: https://pytorch.org/docs/stable/onnx.html#writes-sets
        attn_probs_from_global_key = attn_probs_from_global_key.transpose(1, 3)
        attn_probs_from_global_key[
            is_local_index_no_global_attn_nonzero[0], is_local_index_no_global_attn_nonzero[1], :, :
        ] = torch.finfo(attn_probs_from_global_key.dtype).min
        attn_probs_from_global_key = attn_probs_from_global_key.transpose(1, 3)

        return attn_probs_from_global_key

    def _compute_out_all_to_global(
        self,
        value: torch.Tensor,
        attn_probs: torch.Tensor,
        max_num_global_attn_indices: int,
        is_index_global_attn_nonzero: tuple,
        is_local_index_global_attn_nonzero: tuple,
    ) -> torch.Tensor:
        """
        Compute the attention output of all tokens attending to global.

        Args:
            value (torch.Tensor): (batch, head, time, head_dim) The value vectors for global attention.
            attn_probs (torch.Tensor): (batch, head, time, max_num_global_attn_indices) The attention probabilities.
            max_num_global_attn_indices (int): Maximum number of global attention indices in the batch.
            is_index_global_attn_nonzero (tuple): Indices of global attention (non-zero elements).
            is_local_index_global_attn_nonzero (tuple): Non-padding values within global attention indices.

        Returns:
            torch.Tensor: (batch, time, head x head_dim) The attention output of all tokens attending to global.
        """
        batch_size, h, time, d_k = value.shape[0], value.shape[1], value.shape[2], value.shape[3]

        value = value.transpose(1, 2)

        # get value vectors for global only
        value_vectors_only_global = value.new_zeros(batch_size, max_num_global_attn_indices, h, d_k)
        value_vectors_only_global[is_local_index_global_attn_nonzero] = value[is_index_global_attn_nonzero]

        # compute attn output only global
        out_all_to_global = torch.matmul(attn_probs, value_vectors_only_global.transpose(1, 2)).transpose(1, 2)

        out_all_to_global = out_all_to_global.reshape(batch_size, time, -1)

        return out_all_to_global

    def _compute_out_global_to_all(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        max_num_global_attn_indices: int,
        is_local_index_global_attn_nonzero: tuple,
        is_index_global_attn_nonzero: tuple,
        is_local_index_no_global_attn_nonzero: tuple,
        is_index_masked: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the attention output of global tokens attending to all.

        Args:
            query (torch.Tensor): (batch, head, time, head_dim) The queries for global attention.
            key (torch.Tensor): (batch, head, time, head_dim) The keys for global attention.
            value (torch.Tensor): (batch, head, time, head_dim) The values for global attention.
            max_num_global_attn_indices (int): Maximum number of global attention indices in the batch.
            is_local_index_global_attn_nonzero (tuple): Non-padding values within global attention indices.
            is_index_global_attn_nonzero (tuple): Indices of global attention (non-zero elements).
            is_local_index_no_global_attn_nonzero (tuple): Padding values within global attention indices.
            is_index_masked (torch.Tensor): (batch, time) A boolean tensor indicating if an index is masked.

        Returns:
            global_attn_output (torch.Tensor): (batch, max_num_global_attn_indices, head x head_dim)
            The attention output of global tokens attending to all.
        """

        batch_size = key.shape[0]
        seq_len = key.shape[2]
        h = key.shape[1]
        d_k = key.shape[-1]

        global_k = key.reshape(batch_size * h, -1, d_k)
        global_v = value.reshape(batch_size * h, -1, d_k)

        global_q = query.transpose(1, 2)
        global_q_from_global = global_q.new_zeros(batch_size, max_num_global_attn_indices, h, d_k)
        global_q_from_global[is_local_index_global_attn_nonzero] = global_q[is_index_global_attn_nonzero]
        global_q_from_global = global_q_from_global.transpose(0, 1).reshape(batch_size * h, -1, d_k)

        # compute attn scores
        global_attn_scores = torch.bmm(global_q_from_global, global_k.transpose(1, 2))
        global_attn_scores = global_attn_scores.view(batch_size, h, max_num_global_attn_indices, seq_len)

        # need to transpose since ONNX export only supports consecutive indexing: https://pytorch.org/docs/stable/onnx.html#writes-sets
        global_attn_scores = global_attn_scores.transpose(1, 2)
        global_attn_scores[
            is_local_index_no_global_attn_nonzero[0], is_local_index_no_global_attn_nonzero[1], :, :
        ] = torch.finfo(global_attn_scores.dtype).min
        global_attn_scores = global_attn_scores.transpose(1, 2)

        # compute global attn probs
        # global_attn_probs = nn.functional.softmax(global_attn_scores, dim=-1)
        is_index_masked = is_index_masked.transpose(2, 3)
        rep_is_index_masked = repeat(is_index_masked, 'b h m t -> b h (m m2) t', m2=global_attn_scores.shape[2])
        global_attn_probs = self.scale_mask_softmax(global_attn_scores, rep_is_index_masked)

        global_attn_probs = global_attn_probs.view(batch_size * h, max_num_global_attn_indices, seq_len)

        if not self.sequence_parallel:
            with tensor_parallel.random.get_cuda_rng_tracker().fork():
                global_attn_probs = self.attention_dropout(global_attn_probs)
        else:
            global_attn_probs = self.attention_dropout(global_attn_probs)

        # global attn output
        global_attn_output = torch.bmm(global_attn_probs, global_v)
        global_attn_output = global_attn_output.view(batch_size, h, max_num_global_attn_indices, d_k)

        global_attn_output = global_attn_output[
            is_local_index_global_attn_nonzero[0], :, is_local_index_global_attn_nonzero[1]
        ]

        global_attn_output = global_attn_output.reshape(global_attn_output.shape[0], -1)

        return global_attn_output

    # Longformer implementation for overlap case
    #
    def _skew(self, x: torch.Tensor, direction: List[int], padding_value: float) -> torch.Tensor:
        """Convert diagonals into columns (or columns into diagonals depending on `direction`

        Args:
            x (torch.Tensor): (batch x head, chunk_count, 2w, 2w)
            direction (List[int]): padding directions
            padding_value (float): value to pad with

        Returns:
            output (torch.Tensor): (batch x head, chunk_count, 2w, 2w + 1)

        """
        x_padded = F.pad(x, direction, value=padding_value)
        x_padded = x_padded.view(*x_padded.size()[:-2], x_padded.size(-1), x_padded.size(-2))
        return x_padded

    def _skew2(self, x: torch.Tensor, padding_value: float) -> torch.Tensor:
        """Shift every row 1 step to right converting columns into diagonals

        Args:
            x (torch.Tensor): (batch x head, chunks_count + 1, w, 2w + 1)
            padding_value (float): value to pad with

        Returns:
            output (torch.Tensor): (batch x head, chunks_count + 1, w, 3w)
        """
        # X = B x C x M x L
        B, C, M, L = x.size()
        x = F.pad(x, (0, M + 1), value=padding_value)  # B x C x M x (L+M+1)
        x = x.view(B, C, -1)  # B x C x ML+MM+M
        x = x[:, :, :-M]  # B x C x ML+MM
        x = x.view(B, C, M, M + L)  # B x C, M x L+M
        x = x[:, :, :, :-1]
        return x

    def _chunk_overlap(self, x: torch.Tensor, w: int) -> torch.Tensor:
        """Convert into overlapping chunks.

        Args:
            x (torch.Tensor): # (batch x head, time, size)
            w (int): Chunk overlap size

        Returns:
            output (torch.Tensor): # (batch x head, chunk_count, 2w, size)
        """

        # non-overlapping chunks of size = 2w
        x = x.view(x.size(0), x.size(1) // (w * 2), w * 2, x.size(2))

        # use `as_strided` to make the chunks overlap with an overlap size = w
        chunk_size = list(x.size())
        chunk_size[1] = chunk_size[1] * 2 - 1

        chunk_stride = list(x.stride())
        chunk_stride[1] = chunk_stride[1] // 2
        return x.as_strided(size=chunk_size, stride=chunk_stride)

    @lru_cache()
    def _get_invalid_locations_mask(self, w: int, device: str):

        diagonals_list = []
        for j in range(-w, 1):
            diagonal_mask = torch.zeros(w, device='cpu', dtype=torch.uint8)
            diagonal_mask[:-j] = 1
            diagonals_list.append(diagonal_mask)

        mask = torch.stack(diagonals_list, dim=-1)
        mask = mask[None, None, :, :]

        ending_mask = mask.flip(dims=(2, 3)).bool().to(device)
        return mask.bool().to(device), ending_mask

    def mask_invalid_locations(
        self, input_tensor: torch.Tensor, w: int,
    ):
        """
        Mask locations invalid for the sliding window attention

        Args:
            input_tensor (torch.Tensor): # (batch x head, time, size)
            w (int): Chunk overlap size
        """
        beginning_mask, ending_mask = self._get_invalid_locations_mask(w, input_tensor.device)
        seq_len = input_tensor.size(2)
        beginning_input = input_tensor[:, :, :w, : w + 1]
        beginning_mask = beginning_mask[:, :, :seq_len].expand(beginning_input.size())
        beginning_input.masked_fill_(beginning_mask, -float('inf'))

        ending_input = input_tensor[:, :, -w:, -(w + 1) :]
        ending_mask = ending_mask[:, :, -seq_len:].expand(ending_input.size())
        ending_input.masked_fill_(ending_mask, -float('inf'))

    def sliding_chunks_matmul_qk(self, q: torch.Tensor, k: torch.Tensor, w: int, padding_value: float) -> torch.Tensor:
        """Matrix multiplication of query x key tensors using with a sliding window attention pattern.
        This implementation splits the input into overlapping chunks of size 2w
        with an overlap of size w

        Args:
            q (torch.Tensor): (batch, head, time, size)
            k (torch.Tensor): (batch, head, time, size)
            w (int): Chunk overlap size
            padding_value (float): Value to pad with

        Returns:
            output (torch.Tensor): (batch, head, time, 2w + 1)
        """
        bsz, num_heads, seqlen, head_dim = q.size()
        assert seqlen % (w * 2) == 0
        assert q.size() == k.size()

        chunks_count = seqlen // w - 1

        # group bsz and num_heads dimensions into one, then chunk seqlen into chunks of size w * 2
        q = q.reshape(bsz * num_heads, seqlen, head_dim)
        k = k.reshape(bsz * num_heads, seqlen, head_dim)

        chunk_q = self._chunk_overlap(q, w)  # (batch x head, chunk_count, 2w, size)
        chunk_k = self._chunk_overlap(k, w)  # (batch x head, chunk_count, 2w, size)

        # matrix multipication
        # bcxd: bsz*num_heads x chunks x 2w x head_dim
        # bcyd: bsz*num_heads x chunks x 2w x head_dim
        # bcxy: bsz*num_heads x chunks x 2w x 2w
        chunk_attn = torch.einsum('bcxd,bcyd->bcxy', (chunk_q, chunk_k))  # multiply
        # (batch x head, chunk_count, 2w, 2w)

        # convert diagonals into columns
        diagonal_chunk_attn = self._skew(chunk_attn, direction=(0, 0, 0, 1), padding_value=padding_value)
        # (batch x head, chunk_count, 2w, 2w + 1)

        # allocate space for the overall attention matrix where the chunks are combined. The last dimension
        # has (w * 2 + 1) columns. The first (w) columns are the w lower triangles (attention from a word to
        # w previous words). The following column is attention score from each word to itself, then
        # followed by w columns for the upper triangle.

        diagonal_attn = diagonal_chunk_attn.new_empty((bsz * num_heads, chunks_count + 1, w, w * 2 + 1))
        # (batch x head, chunk_count + 1, w, 2w + 1)

        # copy parts from diagonal_chunk_attn into the compined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_attn[:, :-1, :, w:] = diagonal_chunk_attn[:, :, :w, : w + 1]
        diagonal_attn[:, -1, :, w:] = diagonal_chunk_attn[:, -1, w:, : w + 1]
        # - copying the lower triangle
        diagonal_attn[:, 1:, :, :w] = diagonal_chunk_attn[:, :, -(w + 1) : -1, w + 1 :]
        diagonal_attn[:, 0, 1:w, 1:w] = diagonal_chunk_attn[:, 0, : w - 1, 1 - w :]

        # separate bsz and num_heads dimensions again
        diagonal_attn = diagonal_attn.view(bsz, num_heads, seqlen, 2 * w + 1)
        # (batch, head, time, 2w + 1)

        self.mask_invalid_locations(diagonal_attn, w)

        return diagonal_attn

    def sliding_chunks_matmul_pv(self, prob: torch.Tensor, v: torch.Tensor, w: int):
        """Same as sliding_chunks_matmul_qk but for prob and value tensors.

        Args:
            prob (torch.Tensor): (batch, head, time, size)
            v (torch.Tensor): (batch, head, time, size)
            w (int): Chunk overlap size

        Returns:
            output (torch.Tensor): (batch, time, head, size)
        """
        bsz, num_heads, seqlen, head_dim = v.size()
        chunks_count = seqlen // w - 1
        # group bsz and num_heads dimensions into one, then chunk seqlen into chunks of size 2w
        chunk_prob = prob.reshape(bsz * num_heads, seqlen // w, w, 2 * w + 1)
        # (batch x head, chunks_count + 1, w, 2w + 1)

        # group bsz and num_heads dimensions into one
        v = v.reshape(bsz * num_heads, seqlen, head_dim)
        # (batch x head, time, size)

        # pad seqlen with w at the beginning of the sequence and another w at the end
        padded_v = F.pad(v, (0, 0, w, w), value=-1)
        # (batch x head, time + 2w, size)

        # chunk padded_v into chunks of size 3w and an overlap of size w
        chunk_v_size = (bsz * num_heads, chunks_count + 1, 3 * w, head_dim)
        chunk_v_stride = padded_v.stride()
        chunk_v_stride = chunk_v_stride[0], w * chunk_v_stride[1], chunk_v_stride[1], chunk_v_stride[2]
        chunk_v = padded_v.as_strided(size=chunk_v_size, stride=chunk_v_stride)
        # (batch x head, chunks_count + 1, 3w, size)

        skewed_prob = self._skew2(chunk_prob, padding_value=0)
        # (batch x head, chunks_count + 1, w, 3w)

        context = torch.einsum('bcwd,bcdh->bcwh', (skewed_prob, chunk_v))
        # (batch x head, chunks_count + 1, w, size)

        return context.view(bsz, num_heads, seqlen, head_dim).transpose(1, 2)

# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

"""Utilities for models."""
import itertools
import math
from typing import Dict, Iterator, List, Tuple, Union

import torch
import torch.nn as nn

try:
    from apex.normalization import MixedFusedRMSNorm
    from apex.normalization.fused_layer_norm import FusedLayerNorm  # NOQA
    from apex.transformer.enums import AttnMaskType
    from apex.transformer.layers.layer_norm import FastLayerNorm
    from apex.transformer.pipeline_parallel.schedules.common import listify_model

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

try:
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.tensor_parallel.layers import linear_with_grad_accumulation_and_async_allreduce

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


class ApexGuardDefaults(object):
    """
    This class can be used to replace missing classes when apex is missing.
    """

    def __init__(self):
        super().__init__()

    def __getattr__(self, item):
        return None


def parallel_lm_logits(
    input_: torch.Tensor,
    word_embeddings_weight: torch.Tensor,
    parallel_output: bool,
    bias: torch.Tensor = None,
    async_tensor_model_parallel_allreduce: bool = False,
    sequence_parallel: bool = False,
    gradient_accumulation_fusion: bool = False,
):
    """Language Model logits using word embedding weights.

    Args:
        input_ (torch.Tensor): [b, s, h]
        word_embeddings_weight (torch.Tensor): [(padded) vocab size, h]
        parallel_output (bool): False will gather logits from tensor model parallel region
        bias (torch.Tensor, optional): bias tensor. Defaults to None.
        async_tensor_model_parallel_allreduce (bool, optional): TODO: understand this flag. Defaults to False.
        sequence_parallel (bool, optional): If True will use sequence parallelism. Defaults to False.
        gradient_accumulation_fusioa (bool, optional): If True fuse gradient accumulation to WGRAD GEMM

    Returns:
        torch.Tensor: [b, s, (padded) vocab size]
    """

    tensor_model_parallel = parallel_state.get_tensor_model_parallel_world_size() > 1

    # async grad allreduce can only be used when not using sequence parallelism
    async_grad_allreduce = async_tensor_model_parallel_allreduce and tensor_model_parallel and not sequence_parallel

    # copy input_ to model parallel region if needed
    if async_tensor_model_parallel_allreduce or sequence_parallel:
        input_parallel = input_

    else:
        input_parallel = tensor_parallel.copy_to_tensor_model_parallel_region(input_)

    # Matrix multiply.
    logits_parallel = linear_with_grad_accumulation_and_async_allreduce(
        input=input_parallel,
        weight=word_embeddings_weight,
        bias=bias,
        gradient_accumulation_fusion=gradient_accumulation_fusion,
        async_grad_allreduce=async_grad_allreduce,
        sequence_parallel_enabled=sequence_parallel,
    )

    # Gather if needed.
    if parallel_output:
        return logits_parallel
    else:
        return tensor_parallel.gather_from_tensor_model_parallel_region(logits_parallel)


def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def init_method_const(val):
    def init_(tensor):
        return torch.nn.init.constant_(tensor, val)

    return init_


def scaled_init_method_normal(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores


def get_linear_layer(rows, columns, init_method):
    """Simple linear layer with weight initialization."""
    layer = torch.nn.Linear(rows, columns)
    init_method(layer.weight)
    with torch.no_grad():
        layer.bias.zero_()
    return layer


@torch.jit.script
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))


def openai_gelu(x):
    return gelu_impl(x)


def squared_relu(x):
    return torch.pow(torch.nn.functional.relu(x), 2)


# This is actually Python equivalent of torch.nn.functional.gelu(), also with type hints for ONNX exporter
@torch.jit.script
def erf_gelu(x):
    return x * 0.5 * (torch.erf(x / 1.41421).to(dtype=x.dtype) + torch.ones_like(x).to(dtype=x.dtype))


def average_losses_across_data_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat([loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses, group=parallel_state.get_data_parallel_group())
    averaged_losses = averaged_losses / torch.distributed.get_world_size(
        group=parallel_state.get_data_parallel_group()
    )

    return averaged_losses


def get_ltor_masks_and_position_ids(data, eod_token, reset_position_ids, reset_attention_mask, eod_mask_loss):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)).view(
        att_mask_batch, 1, seq_length, seq_length
    )

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).repeat(micro_batch_size, 1)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indicies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1) :, : (i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1) :] -= i + 1 - prev_index
                    prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids


def attn_mask_postprocess(attn_mask):
    # [b, 1, s, s]
    # Attn_masks for enc-dec attn and dec attn is None when trying to get just the encoder hidden states.
    if attn_mask is None:
        return None
    extended_attention_mask = attn_mask.unsqueeze(1)
    return extended_attention_mask


def enc_dec_extended_attention_mask(attention_mask_list):

    return [attn_mask_postprocess(attn_mask) for attn_mask in attention_mask_list]


def build_position_ids(token_ids):
    # Create position ids
    seq_length = token_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=token_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(token_ids).clone()

    return position_ids


def make_attention_mask_3d(source_mask, target_mask):
    """
    Returns a 3-dimensional (3-D) attention mask
    :param source_block: 2-D array
    :param target_block: 2-D array
    """
    mask = target_mask[:, None, :] * source_mask[:, :, None]
    return mask


def make_inference_attention_mask_3d(source_block, target_block, pad_id):
    """
    Returns a 3-dimensional (3-D) attention mask
    :param source_block: 2-D array
    :param target_block: 2-D array
    """
    # mask = (target_block[:, None, :] != pad_id) * (source_block[:, :, None] != pad_id)
    return make_attention_mask_3d(source_block != pad_id, target_block != pad_id)


def make_inference_history_mask_3d(block):
    batch, length = block.shape
    arange = torch.arange(length, device=block.device)
    history_mask = (arange[None,] <= arange[:, None])[
        None,
    ]
    history_mask = history_mask.expand(batch, length, length)
    return history_mask


def build_attention_mask_3d_padding(source_mask, target_mask):
    """
    Returns a 3D joint attention mask for Megatron given two 2D masks
    :param source_mask - True for non-masked, else masked [batch, src length]
    :param target_mask - True for non-masked, else masked [batch, tgt length]
    """
    mask = make_attention_mask_3d(source_mask, target_mask)
    # invert mask for Megatron
    return mask < 0.5


def build_attention_mask_3d_causal(source_mask, target_mask):
    """
    Returns a 3D joint attention mask for Megatron given two 2D masks
    :param source_mask - True for non-masked, else masked [batch, src length]
    :param target_mask - True for non-masked, else masked [batch, tgt length]
    """
    causal_mask = make_inference_history_mask_3d(target_mask)
    mask = make_attention_mask_3d(source_mask, target_mask)
    mask = mask * causal_mask
    # invert mask for Megatron
    return mask < 0.5


def build_attention_mask_3d(source_mask, target_mask, attn_mask_type):
    """
    Returns a 3D attention mask for Megatron given two 2D masks
    :param source_mask - < 0.5 for non-masked, else masked [batch, src length]
    :param target_mask - < 0.5 for non-masked, else masked [batch, tgt length]
    :param attn_mask_type - AttnMaskType enum
    """
    if attn_mask_type == AttnMaskType.padding:
        mask = build_attention_mask_3d_padding(source_mask, target_mask)
    elif attn_mask_type == AttnMaskType.causal:
        mask = build_attention_mask_3d_causal(source_mask, target_mask)
    else:
        raise ValueError(f"Unsupported attention mask attn_mask_type = {attn_mask_type}")

    return mask


def get_params_for_weight_decay_optimization(
    model: Union[torch.nn.Module, List[torch.nn.Module]],
) -> Dict[str, torch.nn.Parameter]:
    """Divide params into with-weight-decay and without-weight-decay groups.

    Layernorms and biases will have no weight decay but the rest will.
    """
    modules = listify_model(model)
    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
    for module in modules:
        for module_ in module.modules():
            if isinstance(module_, (FusedLayerNorm, FastLayerNorm, MixedFusedRMSNorm)):
                no_weight_decay_params['params'].extend(
                    [p for p in list(module_._parameters.values()) if p is not None]
                )
            else:
                weight_decay_params['params'].extend(
                    [p for n, p in list(module_._parameters.items()) if p is not None and n != 'bias']
                )
                no_weight_decay_params['params'].extend(
                    [p for n, p in list(module_._parameters.items()) if p is not None and n == 'bias']
                )

    return weight_decay_params, no_weight_decay_params


def get_all_params_for_weight_decay_optimization(
    model: Union[torch.nn.Module, List[torch.nn.Module]],
) -> Tuple[Dict[str, List[torch.nn.Parameter]]]:
    """Use all params for weight decay."""
    modules = listify_model(model)

    weight_decay_params = [
        p for module in modules for module_ in module.modules() for p in module_._parameters.values() if p is not None
    ]

    return ({'params': weight_decay_params},)


def get_iterator_k_split(batch: List[torch.Tensor], num_microbatches: int) -> Iterator:
    if isinstance(batch, dict):
        items = list(batch.items())
        assert items[0][1].shape[0] % num_microbatches == 0, "Issue with batch size configuration!"
        split_batch = [torch.tensor_split(item[1], num_microbatches, dim=0) for item in items]
        microbatches = [[(items[i][0], split_batch[i][j]) for i in range(len(items))] for j in range(num_microbatches)]
        microbatches = [dict(elem) for elem in microbatches]
    else:
        assert batch[0].shape[0] % num_microbatches == 0, "Issue with batch size configuration!"
        split_batch = [torch.tensor_split(item, num_microbatches, dim=0) for item in batch]
        microbatches = [[elem[i] for elem in split_batch] for i in range(num_microbatches)]

    return itertools.chain(microbatches)


def make_global_fixed_block_ids(
    attention_mask: torch.Tensor, global_block_size: int
):
    """
    Adapted from HuggingFace LongT5 Model
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/longt5/modeling_longt5.py#L153
    """
    batch_size, seq_len = attention_mask.shape[:2]

    def handle_orphan_tokens(block_ids: torch.Tensor) -> torch.Tensor:
        block_ends = (torch.arange(seq_len) % global_block_size) == global_block_size - 1
        block_ends = block_ends.to(block_ids.device)
        true_block_ends = torch.logical_and(block_ends, block_ids >= 0)
        full_blocks = true_block_ends.sum(-1).unsqueeze(-1).type(block_ids.dtype) - 1
        block_ids = torch.where(block_ids < full_blocks, block_ids, full_blocks)
        return block_ids

    fixed_block_mask = torch.ones_like(attention_mask, device=attention_mask.device) / global_block_size
    fixed_block_mask = torch.cumsum(fixed_block_mask, axis=1) - fixed_block_mask
    mask = torch.where(attention_mask != 0.0, 1.0, -1000.0).type(attention_mask.dtype)
    global_block_ids = torch.floor(mask + fixed_block_mask - 1.0).type(attention_mask.dtype)
    _global_block_ids_lower_bound = torch.tensor(-1, dtype=global_block_ids.dtype, device=global_block_ids.device)
    global_block_ids = torch.where(
        global_block_ids > _global_block_ids_lower_bound, global_block_ids, _global_block_ids_lower_bound
    )
    # set padding tokens to -1
    global_block_ids = (global_block_ids * attention_mask) + (attention_mask - 1)
    # [batch_size, seq_len]
    global_block_ids = handle_orphan_tokens(global_block_ids)
    num_globals = seq_len // global_block_size
    # [batch_size, seq_len // global_block_size]
    if num_globals > 0:
        _sequence_block_ids_max = torch.max(global_block_ids, dim=-1).values.repeat(num_globals, 1).transpose(0, 1)
    else:
        _sequence_block_ids_max = torch.zeros(
            batch_size, 0, dtype=global_block_ids.dtype, device=global_block_ids.device
        )
    global_segment_ids = torch.cumsum(torch.ones(batch_size, num_globals), dim=-1) - 1
    global_segment_ids = global_segment_ids.to(attention_mask.device)
    global_segment_ids = torch.where(global_segment_ids <= _sequence_block_ids_max, 1, 0)
    return global_block_ids.type(torch.int), global_segment_ids.type(torch.int)


def pad_to_multiple(x: torch.Tensor, block_len: int, dim: int, pad_value: int = 0) -> torch.Tensor:
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


def split_into_blocks(x: torch.Tensor, block_len: int, dim: int) -> torch.Tensor:
    """Split an input tensor into blocks of a given `block_len` along the given `dim`. If the dimension length
    is not a multiple of `block_len`, it will be padded first with selected `pad_value`.
    """
    # pad tensor to multiple of block_len
    if x.shape[dim] % block_len != 0:
        x = pad_to_multiple(x, block_len, dim, pad_value=0)
    num_blocks = x.shape[dim] // block_len
    output_shape = x.shape[:dim] + (num_blocks, block_len) + x.shape[(dim + 1) :]
    # If 0 is in output_shape, we cannot apply reshape because of incompatibility with ONNX conversion
    if 0 in output_shape:
        return torch.empty(output_shape, dtype=x.dtype, device=x.device)
    return x.reshape(output_shape)


def concatenate_3_blocks(x: torch.Tensor, block_dim: int, sequence_dim: int, pad_value: int = 0) -> torch.Tensor:
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


def create_global_aggregates(
    hidden_states: torch.Tensor, block_ids: torch.Tensor, global_seq_len: int
):
    """Compute individual block aggregates by summing over individual blocks."""
    # (batch..., seq_len, global_seq_len))
    block_ids = block_ids.where(
        block_ids >= 0, torch.tensor(global_seq_len, dtype=block_ids.dtype, device=block_ids.device)
    )
    one_hot_block_ids = nn.functional.one_hot(block_ids.type(torch.int64), global_seq_len + 1)[:, :, :-1]
    return torch.einsum("...nd,...ng->...gd", hidden_states, one_hot_block_ids.type(hidden_states.dtype))


def mask_local_attention_mask(local_attention_mask: torch.Tensor, block_len: int) -> torch.Tensor:
    """Mask local attention mask to enforce that tokens are not allowed to attend tokens farther than ``local_context."""
    position_ids = torch.arange(3 * block_len, dtype=torch.int32)
    center_position_ids = position_ids[block_len:-block_len]
    # [block_len, 3 * block_len]
    relative_position_ids = position_ids.unsqueeze(0) - center_position_ids.unsqueeze(1)
    locality_mask = torch.abs(relative_position_ids) < block_len
    locality_mask = locality_mask[None, None, :, :]
    locality_mask = locality_mask.to(local_attention_mask.device)
    return torch.logical_and(local_attention_mask, locality_mask)


def get_local_attention_mask(attention_mask: torch.Tensor, block_len: int, device: torch.device) -> torch.Tensor:
    """Prepare attention mask to be applied for a local attention."""
    # [batch_size, num_blocks, block_len]
    _blocked_attention_mask = split_into_blocks(attention_mask, block_len, dim=1)
    # [batch_size, num_block, 3 * block_len]
    _3blocked_attention_mask = concatenate_3_blocks(_blocked_attention_mask, block_dim=1, sequence_dim=2)

    _blocked_attention_mask = _blocked_attention_mask.unsqueeze(-1)
    _3blocked_attention_mask = _3blocked_attention_mask.unsqueeze(-2)
    # [batch_size, num_block, block_len, 3 * block_len]
    local_attention_mask = torch.logical_and(_blocked_attention_mask, _3blocked_attention_mask)
    local_attention_mask = mask_local_attention_mask(local_attention_mask, block_len)
    # [batch_size, num_block, 1, block_len, 3 * block_len]
    return local_attention_mask.unsqueeze(1).to(device).transpose(1, 2)
import numpy as np

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class BlockdiagButterflyMultiply(torch.autograd.Function):
    """This is a faster implementation, with careful memory copies for the fastest
    bmm performance.
    The backward pass is also written manually with careful memory copies.
    Arguments:
        x: (batch, n)
        w1_bfly: (k, q, p), where k = n / p
        w2_bfly: (l, s, r), where l = k * q / r = n * q / (p * r)
    Outputs:
        out: (batch, m), where m = l * s = n * s * q / (p * r)
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, x, w1_bfly, w2_bfly):
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        k, q, p = w1_bfly.shape
        l, s, r = w2_bfly.shape
        assert k * p == n
        assert l * r == k * q
        x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
        out1 = torch.empty(batch_dim, k, q, device=x.device, dtype=x.dtype).transpose(0, 1)
        out1 = torch.bmm(x_reshaped, w1_bfly.transpose(-1, -2), out=out1)
        out1 = out1.transpose(0, 1).reshape(batch_dim, r, l).transpose(-1, -2).contiguous().transpose(0, 1)
        out2 = torch.empty(batch_dim, l, s, device=x.device, dtype=x.dtype).transpose(0, 1)
        out2 = torch.bmm(out1, w2_bfly.transpose(-1, -2), out=out2)
        out2 = out2.permute(1, 2, 0).reshape(*batch_shape, s * l)
        ctx.save_for_backward(x, w1_bfly, w2_bfly, out1)
        return out2

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout):
        x, w1_bfly, w2_bfly, out1 = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        k, q, p = w1_bfly.shape
        l, s, r = w2_bfly.shape
        # assert k * p == n
        # assert l * r == k * q
        dx, dw1_bfly, dw2_bfly = None, None, None
        # dout_reshaped = dout.reshape(batch_dim, sqrtn, sqrtn).permute(2, 1, 0).contiguous()
        dout_reshaped = dout.reshape(batch_dim, s, l).transpose(-1, -2).contiguous()
        dout_reshaped = dout_reshaped.transpose(0, 1)
        if ctx.needs_input_grad[2]:
            # dw2_bfly = torch.empty(l, s, r, device=w2_bfly.device, dtype=w2_bfly.dtype)
            # dw2_bfly = torch.bmm(dout_reshaped.transpose(-1, -2), out1, out=dw2_bfly)
            dw2_bfly = torch.bmm(dout_reshaped.transpose(-1, -2), out1.conj())
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[0]:
            dout1 = torch.empty(batch_dim, l, r, device=x.device, dtype=x.dtype).transpose(0, 1)
            dout1 = torch.bmm(dout_reshaped, w2_bfly.conj(), out=dout1)
            dout1 = dout1.transpose(0, 1).transpose(-1, -2).contiguous().reshape(batch_dim, k, q).transpose(0, 1)
            # dout1 = dout1.permute(1, 2, 0).contiguous().transpose(0, 1)
            if ctx.needs_input_grad[0]:
                dx = torch.empty(batch_dim, k, p, device=x.device, dtype=x.dtype)
                dx = torch.bmm(dout1, w1_bfly.conj(), out=dx.transpose(0, 1)).transpose(0, 1).reshape(*batch_shape, n)
            if ctx.needs_input_grad[1]:
                x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
                dw1_bfly = torch.bmm(dout1.transpose(-1, -2), x_reshaped.conj())
        return dx, dw1_bfly, dw2_bfly


blockdiag_butterfly_multiply = BlockdiagButterflyMultiply.apply


class StructuredLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        """Subclasses should call reset_parameters
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Subclasses may override {in,out}_features_extended
        if not hasattr(self, 'in_features_extended'):
            self.in_features_extended = in_features
        if not hasattr(self, 'out_features_extended'):
            self.out_features_extended = out_features
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self) -> None:
        self.set_weights_from_dense_init(dense_init_fn_=partial(init.kaiming_uniform_, a=math.sqrt(5)))
        self.reset_parameters_bias()

    def set_weights_from_dense_init(self, dense_init_fn_):
        raise NotImplementedError

    def reset_parameters_bias(self):
        if self.bias is not None:
            fan_in = self.bias.shape[-1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    @property
    def saving(self):
        raise NotImplementedError

    def convert_to_dense_weight(self):
        factory_kwargs = {'device': self.weight.device, 'dtype': self.weight.dtype}
        dense_weight = self.forward_matmul(torch.eye(self.in_features, **factory_kwargs)).T
        return dense_weight

    def preprocess(self, x):
        in_features = x.shape[-1]
        if in_features < self.in_features_extended:
            x = F.pad(x, (0, self.in_features_extended - in_features))
        return x

    def postprocess(self, output):
        out_features_extended = output.shape[-1]
        if out_features_extended > self.out_features:
            output = output[..., :self.out_features]
        return output

    def forward_matmul(self, x):
        raise NotImplementedError

    def forward(self, x):
        output = self.forward_matmul(x)
        # Convert bias to output.dtype in case of AMP, otherwise bias and activation will be in FP32
        return (output + self.bias.to(dtype=output.dtype)) if self.bias is not None else output


class MonarchLinear(StructuredLinear):

    def __init__(self, *args, nblocks=4, **kwargs):
        super().__init__(*args, **kwargs)
        in_blksz = int(math.ceil(self.in_features / nblocks))
        out_blksz = int(math.ceil(self.out_features / nblocks))
        self.in_features_extended = in_blksz * nblocks
        self.out_features_extended = out_blksz * nblocks

        if self.in_features_extended < self.out_features_extended:
            self.blkdiag1 = nn.Parameter(torch.empty(nblocks, in_blksz, in_blksz))
            self.blkdiag2 = nn.Parameter(torch.empty(nblocks, out_blksz, in_blksz))
        else:
            self.blkdiag1 = nn.Parameter(torch.empty(nblocks, out_blksz, in_blksz))
            self.blkdiag2 = nn.Parameter(torch.empty(nblocks, out_blksz, out_blksz))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Mimic init.kaiming_uniform: https://github.com/pytorch/pytorch/blob/24087d07ca7ffa244575d259711dd7c99245a67a/torch/nn/init.py#L360
        for blkdiag in [self.blkdiag1, self.blkdiag2]:
            fan_in = blkdiag.shape[-1]
            gain = init.calculate_gain(nonlinearity='leaky_relu', param=math.sqrt(5))
            std = gain / math.sqrt(fan_in)
            bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            with torch.no_grad():
                blkdiag.uniform_(-bound, bound)
        self.reset_parameters_bias()

    def forward_matmul(self, x):
        output = blockdiag_butterfly_multiply(self.preprocess(x), self.blkdiag1, self.blkdiag2)
        return self.postprocess(output)

@functools.lru_cache
def get_butterfly_indices(
    out_features: int,
    in_features: int,
    block_size: int = 256,
    butterfly_size: int = 64,
    n_factors: Optional[int] = None,
    stretch: bool = False,
) -> torch.IntTensor:
    """
    Get a matrix [num_output_blocks, num_active_input_blocks] with int32 indices for additive butterfly
    Based on the original implementation from https://arxiv.org/abs/2112.00029 .

    :param stretch: by default, non-square matrices will have stretched butterfly patterns,
      otherwise the square pattern will be repeated a given number of times
    """
    assert (
        out_features % in_features == 0 or in_features % out_features == 0
    ), "if matrix is not square, the longer dimension must be a multiple of the shorter dimension"
    assert out_features % block_size == 0 and in_features % block_size == 0
    log_n = int(math.log2(butterfly_size))
    n_factors = log_n if n_factors is None else n_factors
    if butterfly_size != 2 ** log_n or butterfly_size < 2:
        raise NotImplementedError("butterfly_size must be a power of 2")
    if not (1 <= n_factors <= log_n):
        raise NotImplementedError(
            "n_factors must be a between 1 and log_2(butterfly_size)"
        )

    twiddle = torch.ones(butterfly_size // 2, 2, 2)
    layout = sum(
        butterfly_factor_to_matrix(twiddle, index) for index in range(n_factors)
    )
    layout = layout.bool().int()
    # Convert from (butterfly_size, butterfly_size) mask to (out_features, in_features) mask
    layout = einops.repeat(
        layout,
        "b b1 -> (b f) (b1 f1)",
        f=out_features // butterfly_size,
        f1=in_features // butterfly_size,
    )
    # Convert from (out_features, in_features) mask to
    # (out_features // block_size, in_features // block_size) mask
    layout = einops.rearrange(
        layout,
        "(p blksz) (r blksz1) -> p r (blksz blksz1)",
        blksz=block_size,
        blksz1=block_size,
    )

    layout = (layout > 0).any(
        dim=-1
    )  # [out_features // block_size, in_features // block_size]
    if not stretch:
        out_blocks, in_blocks = layout.shape
        if out_blocks > in_blocks:
            ratio = out_blocks // in_blocks
            layout = (
                layout.view(out_blocks // ratio, ratio, in_blocks)
                .permute(1, 0, 2)
                .reshape_as(layout)
            )
        elif out_blocks < in_blocks:
            ratio = in_blocks // out_blocks
            layout = (
                layout.view(out_blocks, in_blocks // ratio, ratio)
                .permute(0, 2, 1)
                .reshape_as(layout)
            )

    # convert boolean layout to indices for F.embedding_bag
    num_output_blocks = out_features // block_size
    num_input_blocks = in_features // block_size
    active_blocks_per_output = layout.sum(1).unique()
    assert (
        len(active_blocks_per_output) == 1
    ), "butterfly layout must have the same number of blocks per row"
    active_blocks_per_output = active_blocks_per_output.item()

    active_blocks_per_input = layout.sum(0).unique()
    assert (
        len(active_blocks_per_input) == 1
    ), "butterfly layout must have the same number of blocks per row"
    active_blocks_per_input = active_blocks_per_input.item()

    # which input blocks should be added for i-th output
    input_block_index = layout.nonzero()[:, 1].view(
        num_output_blocks, active_blocks_per_output
    )
    # which output blocks does j-th input contribute to
    output_block_index = (
        layout.t().nonzero()[:, 1].view(num_input_blocks, active_blocks_per_input)
    )

    # which of the active blocks from the corresponding input_block should be used for i-th output
    active_block_index = torch.where(
        torch.eq(
            output_block_index[input_block_index],
            torch.arange(len(input_block_index))[:, None, None],
        )
    )[-1].view(input_block_index.shape)

    return input_block_index * active_blocks_per_input + active_block_index


def butterfly_factor_to_matrix(
    twiddle: torch.Tensor, factor_index: int
) -> torch.Tensor:
    """
    Let b be the base (most commonly 2).
    Parameters:
        twiddle: (n // b, b, b)
        factor_index: an int from 0 to log_b(n) - 1
    """
    n_div_b, b, _ = twiddle.shape
    n = b * n_div_b
    log_b_n = int(math.log(n) / math.log(b))
    assert n == b ** log_b_n, f"n must be a power of {b}"
    assert twiddle.shape == (n // b, b, b)
    assert 0 <= factor_index <= log_b_n
    stride = b ** factor_index
    x = einops.rearrange(
        torch.eye(n), "bs (diagblk j stride) -> bs diagblk j stride", stride=stride, j=b
    )
    t = einops.rearrange(
        twiddle, "(diagblk stride) i j -> diagblk stride i j", stride=stride
    )
    out = torch.einsum("d s i j, b d j s -> b d i s", t, x)
    out = einops.rearrange(out, "b diagblk i stride -> b (diagblk i stride)")
    return (
        out.t()
    )  # Transpose because we assume the 1st dimension of x is the batch dimension


def butterfly_matmul(
    input: torch.Tensor, weight: torch.Tensor, butterfly_flat_indices: torch.Tensor
):
    """
    :param input: tensor [*batch_dims, in_features]
    :param weight: tensor [in_features, active_blocks_per_input, block_size]
    :param butterfly_flat_indices: outputs of get_butterfly_indices(...)
    :returns: tensor [*batch_dims, out_features]
    """
    assert input.shape[-1] == weight.shape[0]
    in_features, active_blocks_per_input, block_size = weight.shape
    num_input_blocks = in_features // block_size
    batch_dims = input.shape[:-1]
    input = input.flatten(0, -2)

    input_permuted = input.t().view(
        input.shape[1] // block_size, block_size, input.shape[0]
    )
    output_blocks = torch.bmm(
        weight.view(num_input_blocks, -1, block_size), input_permuted
    )
    # ^-- shape: [num_input_blocks, (active_blocks_per_input * block_size), flat_batch_dims]

    blocks_for_indexing = output_blocks.view(
        num_input_blocks * active_blocks_per_input, block_size * input.shape[0]
    )
    # ^-- shape: [(num_input_blocks * active_blocks_per_input),  (block_size, flat_batch_dims)]

    aggregated_blocks = F.embedding_bag(
        butterfly_flat_indices, blocks_for_indexing, mode="sum"
    )
    # ^-- shape: [num_ouput_blocks, (block_size, flat_batch_dims)]

    outputs = aggregated_blocks.view(-1, input.shape[0]).t()
    # ^-- shape: [flat_batch_dims, (num_output_blocks * block_size)] aka [flat_batch_dims, out_features]
    return outputs.view(*batch_dims, outputs.shape[-1])


class PixelflyLinear(nn.Module):
    def __init__(
            self, in_features: int, out_features: int,
            lowrank_size: int, block_size: int, butterfly_size: int,
            n_factors: Optional[int] = None, stretch: bool = True, bias: bool = True,
    ):
        super().__init__()
        self.out_features, self.in_features = out_features, in_features
        self.register_buffer("butterfly_flat_indices", get_butterfly_indices(
            out_features, in_features, block_size, butterfly_size, n_factors, stretch))
        self.lowrank_first = nn.Linear(in_features, lowrank_size, bias=False)
        self.lowrank_second = nn.Linear(lowrank_size, out_features, bias=bias)

        active_blocks_per_input = self.butterfly_flat_indices.numel() // (in_features // block_size)
        self.weight = nn.Parameter(torch.empty(in_features, active_blocks_per_input, block_size))
        nn.init.xavier_normal_(self.weight)

    def forward(self, input):
        output = self.lowrank_second(self.lowrank_first(input))
        output += butterfly_matmul(input, self.weight, self.butterfly_flat_indices)
        return output
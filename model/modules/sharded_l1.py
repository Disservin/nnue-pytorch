"""Paired-input L1 shards and autograd-aware batch reduce-scatter."""

import math
from functools import lru_cache
from itertools import pairwise

import torch
import torch.distributed as dist
import torch.nn.functional as F

from . import stacked_linear
from .feature_transformer.sharded_feature_transformer import FTShardLayout
from .stacked_linear import FactorizedStackedLinear


@lru_cache(maxsize=1)
def _compiled_l1_merge():
    return torch.compile(ShardedL1.merged_parameters, fullgraph=True)


def _add_local_bias(values, bias, indices, count, out_features):
    return values + bias.view(count, out_features)[indices]


class _SumReduceScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, offsets, group):
        world = len(offsets) - 1
        rank = dist.get_rank(group) if world > 1 else 0
        sizes = tuple(offsets[i + 1] - offsets[i] for i in range(world))
        ctx.equal_batch_fast = (
            world > 1 and sizes[0] > 0 and all(n == sizes[0] for n in sizes)
            and dist.get_backend(group) == "nccl"
        )
        if ctx.equal_batch_fast:
            result = values.new_empty((sizes[0], values.shape[1]))
            dist.reduce_scatter_tensor(result, values.contiguous(), group=group)
            ctx.world, ctx.group = world, group
            return result
        max_size = max(1, *sizes)
        # Equal-size blocks permit NCCL reduce_scatter_tensor with uneven local
        # batches. All ranks use the same rank-order offsets from feature gather.
        blocks = values.new_zeros((world, max_size, values.shape[1]))
        for i, size in enumerate(sizes):
            blocks[i, :size].copy_(values[offsets[i] : offsets[i + 1]])
        result = values.new_empty((max_size, values.shape[1]))
        if world == 1:
            result.copy_(blocks[0])
        elif dist.get_backend(group) == "nccl":
            dist.reduce_scatter_tensor(result, blocks.flatten(0, 1), group=group)
        else:
            # Gloo does not universally implement reduce-scatter. This CPU
            # fallback is also useful for correctness tests without GPUs.
            dist.all_reduce(blocks, group=group)
            result.copy_(blocks[rank])
        ctx.sizes, ctx.max_size, ctx.rank, ctx.group = sizes, max_size, rank, group
        return result[: sizes[rank]].clone()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.equal_batch_fast:
            grad = grad_output.new_empty(
                (ctx.world * grad_output.shape[0], grad_output.shape[1])
            )
            dist.all_gather_into_tensor(grad, grad_output.contiguous(), group=ctx.group)
            return grad, None, None
        padded = grad_output.new_zeros((ctx.max_size, grad_output.shape[1]))
        padded[: ctx.sizes[ctx.rank]].copy_(grad_output)
        blocks = [torch.empty_like(padded) for _ in ctx.sizes]
        if len(blocks) == 1:
            blocks[0] = padded
        else:
            dist.all_gather(blocks, padded, group=ctx.group)
        return torch.cat([b[:n] for b, n in zip(blocks, ctx.sizes)]), None, None


def sum_reduce_scatter_batch(
    values: torch.Tensor, offsets: tuple[int, ...], group=None
):
    """Sum global-batch contributions and return this rank's batch segment.

    Backward gathers local output gradients in rank order without scaling.
    Every process must call both forward and backward in the same order.
    """
    world = (
        dist.get_world_size(group)
        if dist.is_available() and dist.is_initialized()
        else 1
    )
    if (
        values.ndim != 2
        or len(offsets) != world + 1
        or offsets[0] != 0
        or offsets[-1] != values.shape[0]
        or any(a > b for a, b in pairwise(offsets))
    ):
        raise ValueError("offsets must partition the global batch in group rank order")
    return _SumReduceScatter.apply(values, offsets, group)


class ShardedL1(FactorizedStackedLinear):
    """Own paired input columns of L1 and replicated L1 biases.

    Input activations and stack indices are in global-batch order. The local
    result follows the same rank offsets as the feature gather. Construct on
    CPU, then move to the target device. Do not wrap this module in DDP: its
    weights belong to different shards, while its biases are replicated.
    """

    def __init__(
        self,
        full_width,
        out_features,
        count,
        quantization,
        *,
        rank,
        world_size,
        group_maps=(),
    ):
        self.layout = FTShardLayout(full_width, world_size, rank)
        seed = int(
            torch.randint(0, 2**63 - 1, (), dtype=torch.int64, device="cpu").item()
        )
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed((seed + rank) % (2**63 - 1))
            super().__init__(
                2 * self.layout.pair_width, out_features, count, quantization, "ls_l1"
            )
            # Preserve the full layer's fan-in initialization distribution.
            with torch.no_grad():
                self.linear.weight.mul_(math.sqrt(self.in_features / full_width))
                torch.random.default_generator.manual_seed(seed)
                first_bias = torch.empty(out_features).uniform_(
                    -1 / math.sqrt(full_width), 1 / math.sqrt(full_width)
                )
                self.linear.bias.copy_(first_bias.repeat(count))
            self.set_group_factors(list(group_maps))

    def get_extra_state(self):
        return {
            "version": 1,
            "full_width": self.layout.full_width,
            "world_size": self.layout.world_size,
            "rank": self.layout.rank,
            "group_maps": [
                getattr(self, f"group_map_{i}").cpu().tolist()
                for i in range(len(self.group_maps))
            ],
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("L1 checkpoint shard layout does not match this rank")

    def sharded_parameters(self):
        return [self.linear.weight, self.factorized_linear.weight, *self.group_factor_w]

    def replicated_bias_parameters(self):
        return [self.linear.bias, self.factorized_linear.bias, *self.group_factor_b]

    @torch.no_grad()
    def load_unsharded(self, full: FactorizedStackedLinear):
        """Copy paired weight columns and the complete replicated bias."""
        if (
            full.in_features != self.layout.full_width
            or full.out_features != self.out_features
            or full.count != self.count
            or full.quantization.config != self.quantization.config
            or len(full.group_maps) != len(self.group_maps)
        ):
            raise ValueError("unsharded L1 configuration does not match")
        for i in range(len(self.group_maps)):
            if not torch.equal(
                getattr(full, f"group_map_{i}").cpu(),
                getattr(self, f"group_map_{i}").cpu(),
            ):
                raise ValueError("unsharded L1 group maps do not match")
        source, target = dict(full.named_parameters()), dict(self.named_parameters())
        if source.keys() != target.keys():
            raise ValueError("unsharded L1 parameters do not match")
        for name, dest in target.items():
            src = source[name]
            expected = (
                (*src.shape[:-1], self.in_features) if src.ndim == 2 else src.shape
            )
            if dest.shape != expected:
                raise ValueError(f"incompatible L1 parameter shape for {name}")
        for name, dest in target.items():
            src = source[name]
            dest.copy_(self.layout.select_columns(src) if src.ndim == 2 else src)

    def merged_parameters(self, fake_quantize_weights=False):
        """Merge the L1 factors before optional fake quantization."""
        weight = self.linear.weight + self.factorized_linear.weight.repeat(
            self.count, 1
        )
        bias = self.linear.bias + self.factorized_linear.bias.repeat(self.count)
        ge = self.group_expansion()
        if ge is not None:
            weight, bias = weight + ge[0], bias + ge[1]
        if fake_quantize_weights:
            weight = self.quantization.fake_quantize_weights(weight, "ls_l1_weight")
            bias = self.quantization.fake_quantize_weights(bias, "ls_l1_bias")
        return weight, bias

    def forward(
        self, x, global_indices, offsets, fake_quantize_weights=False, group=None
    ):
        """Compute global partial L1, reduce-scatter, then add local bias.

        Normalize each rank's loss numerator by the global weight sum (or by
        global sample count for an unweighted mean). Sum replicated bias and
        tail gradients across ranks; never reduce FT or L1 weight-shard gradients.
        """
        if (
            x.ndim != 2
            or x.shape[1] != self.in_features
            or global_indices.shape != (x.shape[0],)
        ):
            raise ValueError(
                "L1 shard inputs must have global batch and local column dimensions"
            )
        world = (
            dist.get_world_size(group)
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        rank = dist.get_rank(group) if world > 1 else 0
        if world != self.layout.world_size or rank != self.layout.rank:
            raise ValueError("L1 shard layout must match its process group")
        if (getattr(self, "_compile_l1_merge", False) and x.device.type == "cuda"
                and not torch.is_autocast_enabled()
                and all(p.device.type == "cuda" and p.dtype == torch.float32
                        for p in self.parameters())):
            weight, bias = _compiled_l1_merge()(self, fake_quantize_weights)
        else:
            weight, bias = self.merged_parameters(fake_quantize_weights)
        if stacked_linear.can_use_grouped_l1(x, weight, bias, self.count):
            # Bias is added exactly once after reduce-scatter. This zero tensor
            # is not a parameter and its gradient is never computed.
            zero_bias = weight.new_zeros(self.count * self.out_features)
            partial = stacked_linear.grouped_l1(
                x, weight, zero_bias, global_indices, self.count
            )
        else:
            partial = self.select_output(F.linear(x, weight), global_indices)
        local = sum_reduce_scatter_batch(partial, offsets, group)
        local_indices = global_indices[offsets[rank] : offsets[rank + 1]]
        return _add_local_bias(local, bias, local_indices, self.count, self.out_features)

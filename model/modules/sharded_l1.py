"""Paired-input L1 shards and autograd-aware batch reduce-scatter."""

import math
from functools import lru_cache

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


def _world(group):
    return (
        dist.get_world_size(group)
        if dist.is_available() and dist.is_initialized()
        else 1
    )


class _SumReduceScatter(torch.autograd.Function):
    """Sum equal-size rank blocks and keep this rank's block.

    Backward all-gathers the block gradients in rank order, unscaled.
    """

    @staticmethod
    def forward(ctx, values, group):
        world = _world(group)
        ctx.world, ctx.group = world, group
        if world == 1:
            return values.clone()
        result = values.new_empty((values.shape[0] // world, values.shape[1]))
        if dist.get_backend(group) == "nccl":
            dist.reduce_scatter_tensor(result, values.contiguous(), group=group)
        else:
            # Gloo has no reduce-scatter; this path serves the CPU tests.
            summed = values.clone()
            dist.all_reduce(summed, group=group)
            result.copy_(summed.chunk(world)[dist.get_rank(group)])
        return result

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.world == 1:
            return grad_output, None
        grad_output = grad_output.contiguous()
        if dist.get_backend(ctx.group) == "nccl":
            grad = grad_output.new_empty(
                (ctx.world * grad_output.shape[0], grad_output.shape[1])
            )
            dist.all_gather_into_tensor(grad, grad_output, group=ctx.group)
            return grad, None
        blocks = [torch.empty_like(grad_output) for _ in range(ctx.world)]
        dist.all_gather(blocks, grad_output, group=ctx.group)
        return torch.cat(blocks), None


def sum_reduce_scatter_batch(values: torch.Tensor, group=None):
    """Sum every rank's partial result and return this rank's batch rows.

    ``values`` holds the global batch in rank order, with equal local
    batches. Every process must call forward and backward in the same order.
    """
    if values.ndim != 2 or values.shape[0] % _world(group):
        raise ValueError("the global batch must split evenly across the ranks")
    return _SumReduceScatter.apply(values, group)


class ShardedL1(FactorizedStackedLinear):
    """Own paired input columns of L1 and replicated L1 biases.

    Input activations and stack indices are in global-batch order, made of
    equal local batches in rank order; the result holds this rank's batch. Construct on
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

    def forward(self, x, global_indices, fake_quantize_weights=False, group=None):
        """Compute global partial L1, reduce-scatter, then add local bias.

        Normalize each rank's loss numerator by the global weight sum. Sum
        replicated bias and tail gradients across ranks; never reduce FT or L1
        weight-shard gradients.
        """
        if (
            x.ndim != 2
            or x.shape[1] != self.in_features
            or global_indices.shape != (x.shape[0],)
        ):
            raise ValueError(
                "L1 shard inputs must have global batch and local column dimensions"
            )
        world = _world(group)
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
        local = sum_reduce_scatter_batch(partial, group)
        local_indices = global_indices.chunk(world)[rank]
        return _add_local_bias(local, bias, local_indices, self.count, self.out_features)

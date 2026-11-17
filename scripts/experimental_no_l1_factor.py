"""Benchmark-only removal of L1's shared factor; group factors remain active.

Apply once to a newly constructed model before optimizer/DDP construction.
The two frozen, zeroed parameters remain as inert checkpoint-compatible
placeholders; their optimizer groups and sharded gradient lists are removed.
This changes the training parameterization and is not a training default.
"""
import torch
import torch.nn.functional as F

from model.modules import stacked_linear
from model.modules.sharded_l1 import ShardedL1, _compiled_l1_merge
from model.modules.stacked_linear import FactorizedStackedLinear


def _merge_without_shared_factor(layer, fake_quantize_weights=False):
    weight, bias = layer.linear.weight, layer.linear.bias
    group = layer.group_expansion()
    if group is not None:
        weight, bias = weight + group[0], bias + group[1]
    if fake_quantize_weights:
        if layer.quantization is None or layer.layer_key is None:
            raise RuntimeError("quantization and layer_key are required")
        weight = layer.quantization.fake_quantize_weights(weight, f"{layer.layer_key}_weight")
        bias = layer.quantization.fake_quantize_weights(bias, f"{layer.layer_key}_bias")
    return weight, bias


def _forward_without_shared_factor(layer, x, indices, fake_quantize_weights=False):
    weight, bias = _merge_without_shared_factor(layer, fake_quantize_weights)
    if (x.ndim == 2 and x.shape[1] == layer.in_features
            and stacked_linear.can_use_grouped_l1(x, weight, bias, layer.count)):
        return stacked_linear.grouped_l1(x, weight, bias, indices, layer.count)
    return layer.select_output(F.linear(x, weight, bias), indices)


def apply_no_shared_l1_factor(nnue):
    """Fold only the shared factors, then remove their training computation."""
    l1 = nnue.model.layer_stacks.l1
    with torch.no_grad():
        l1.linear.weight.add_(l1.factorized_linear.weight.repeat(l1.count, 1))
        l1.linear.bias.add_(l1.factorized_linear.bias.repeat(l1.count))
        l1.factorized_linear.weight.zero_()
        l1.factorized_linear.bias.zero_()
    for p in l1.factorized_linear.parameters():
        p.requires_grad_(False)
        p.grad = None
    for group in nnue.model.weight_clipping:
        if any(p is l1.linear.weight for p in group["params"]):
            group.pop("virtual_params", None)
    FactorizedStackedLinear.forward = _forward_without_shared_factor
    ShardedL1.merged_parameters = _merge_without_shared_factor
    _compiled_l1_merge.cache_clear()

    get_wrapper = nnue.config.optimizer_config.get_optimizer_wrapper

    def get_filtered_wrapper():
        wrapper = get_wrapper()
        configure = wrapper.configure_optimizers

        def configure_trainable(groups):
            filtered = [{**group, "params": [p for p in group["params"] if p.requires_grad]}
                        for group in groups]
            return configure([group for group in filtered if group["params"]])

        wrapper.configure_optimizers = configure_trainable
        return wrapper

    nnue.config.optimizer_config.get_optimizer_wrapper = get_filtered_wrapper

"""Check the benchmark-only shared L1 factor ablation."""
import copy

import pytest
import torch

import model.model as model_module
import model.sharded_model as sharded_model_module
from model.modules.sharded_l1 import ShardedL1, _compiled_l1_merge
from model.modules.stacked_linear import FactorizedStackedLinear
from model.nnue import NNUE
from scripts.experimental_no_l1_factor import apply_no_shared_l1_factor
from test_sharded_adamw import _config
from test_sharded_ft import _Feature


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("quantized", [False, True])
def test_factor_ablation_preserves_initial_outputs_and_group_gradients(sharded, quantized, monkeypatch):
    monkeypatch.setattr(model_module, "get_feature_cls", lambda _: [_Feature])
    monkeypatch.setattr(sharded_model_module, "get_feature_cls", lambda _: [_Feature])
    monkeypatch.setattr(FactorizedStackedLinear, "forward", FactorizedStackedLinear.forward)
    monkeypatch.setattr(ShardedL1, "merged_parameters", ShardedL1.merged_parameters)
    torch.manual_seed(31)
    reference = NNUE(_config(), sharding=(0, 1) if sharded else None)
    with torch.no_grad():
        for name, p in reference.model.layer_stacks.named_parameters():
            if "factorized" in name or "group_factor" in name:
                p.uniform_(-0.05, 0.05)
    candidate = copy.deepcopy(reference)
    x = torch.randn(7, 16)
    indices = torch.arange(7, dtype=torch.int64) % reference.model.layer_stacks.count
    ref_l1 = reference.model.layer_stacks.l1
    output = (ref_l1(x, indices, (0, len(x)), quantized) if sharded
              else ref_l1(x, indices, quantized))
    output.square().sum().backward()
    expected_grads = {name: p.grad.clone() for name, p in ref_l1.named_parameters()
                      if p.grad is not None and not name.startswith("factorized_linear.")}
    expected = output.detach().clone()
    expected_export = None if sharded else ref_l1.at_index(0)
    apply_no_shared_l1_factor(candidate)
    try:
        l1 = candidate.model.layer_stacks.l1
        actual = (l1(x, indices, (0, len(x)), quantized) if sharded
                  else l1(x, indices, quantized))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().sum().backward()
        for name, p in l1.named_parameters():
            if name.startswith("factorized_linear."):
                assert not p.requires_grad and p.grad is None
                assert not torch.count_nonzero(p)
            else:
                torch.testing.assert_close(p.grad, expected_grads[name], rtol=1e-6, atol=1e-6)
        assert "virtual_params" not in candidate.model.weight_clipping[0]
        assert len(l1.group_factor_w) > 0
        if expected_export is not None:
            actual_export = l1.at_index(0)
            torch.testing.assert_close(actual_export.weight, expected_export.weight, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(actual_export.bias, expected_export.bias, rtol=1e-6, atol=1e-6)
        optimizers, _ = candidate.configure_optimizers()
        optimized = {id(p) for group in optimizers[0].param_groups for p in group["params"]}
        assert all(id(p) not in optimized for p in l1.factorized_linear.parameters())
        assert all(id(p) in optimized for p in l1.group_factor_w)
    finally:
        _compiled_l1_merge.cache_clear()

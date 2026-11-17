"""Numerical checks for the experimental compiled clipping and L1 merge."""
import copy
import io
import os

import pytest
import torch

import model.model as model_module
import model.sharded_model as sharded_model_module
from model.modules.sharded_l1 import ShardedL1
from model.nnue import NNUE
from model.quantize import QuantizationConfig, QuantizationManager
from test_sharded_adamw import _config
from test_sharded_ft import _Feature


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("keys,factors", [("queen", ""), ("queen", "queen"),
                                           ("queen:piececount", "queen:piececount")])
def test_compiled_weight_clipping_matches_eager(sharded, keys, factors, monkeypatch):
    monkeypatch.setattr(model_module, "get_feature_cls", lambda _: [_Feature])
    monkeypatch.setattr(sharded_model_module, "get_feature_cls", lambda _: [_Feature])
    torch.manual_seed(203)
    config = _config()
    config.model_config.ls_keys = keys
    config.model_config.ls_factors = factors
    if keys != "queen":
        config.model_config.ls_remap_table = ""
    nnue = NNUE(config, sharding=(0, 1) if sharded else None).cuda()
    with torch.no_grad():
        for name, p in nnue.model.named_parameters():
            if "layer_stacks" in name:
                p.uniform_(-2.5, 2.5)
        for group in nnue.model.weight_clipping:
            p = group["params"][0]
            if "virtual_params" in group and len(group["stacked_layer"].group_factor_w):
                group["virtual_params"][0, 0] = 2**20
                layer = group["stacked_layer"]
                group_row = int(layer.group_map_0[0]) * layer.out_features
                layer.group_factor_w[0][group_row, 0] = -(2**20)
            ge = group["stacked_layer"].group_expansion()
            offsets = torch.zeros_like(p)
            if "virtual_params" in group:
                v = group["virtual_params"]
                offsets = v.repeat(p.shape[0] // v.shape[0], p.shape[1] // v.shape[1])
            group_offsets = torch.zeros_like(p) if ge is None else ge[0]
            lo = (p.new_full(p.shape, group["min_weight"]) - offsets) - group_offsets
            hi = (p.new_full(p.shape, group["max_weight"]) - offsets) - group_offsets
            p.view(-1)[:7] = torch.stack((lo.flatten()[0],
                torch.nextafter(lo.flatten()[1], lo.new_tensor(float("-inf"))),
                hi.flatten()[2], torch.nextafter(hi.flatten()[3], hi.new_tensor(float("inf"))),
                p.new_tensor(float("nan")), p.new_tensor(float("inf")),
                (lo.flatten()[6] + hi.flatten()[6]) / 2))
    candidate = copy.deepcopy(nnue.model)
    candidate._compile_layer_clipping = True
    for include_input in (False, False, True):
        nnue.model.clip_weights(include_input=include_input)
        candidate.clip_weights(include_input=include_input)
        for (name, ref), (actual_name, actual) in zip(
                nnue.model.named_parameters(), candidate.named_parameters()):
            assert name == actual_name
            torch.testing.assert_close(actual, ref, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("maps", [[], [[0, 0, 1, 1, 2, 2]],
                                    [[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]]])
@pytest.mark.parametrize("quantized", [False, True])
def test_compiled_l1_merge_outputs_and_gradients(maps, quantized):
    torch.manual_seed(919)
    quant = QuantizationManager(QuantizationConfig())
    layer = ShardedL1(64, 8, 6, quant, rank=0, world_size=1,
                      group_maps=[torch.tensor(m) for m in maps]).cuda()
    with torch.no_grad():
        for p in layer.parameters():
            p.uniform_(-0.1, 0.1)
        # Include exact half-step and adjacent values in the merged weight.
        layer.linear.weight[:3, :1] = torch.tensor(
            [0.5 / 128, torch.nextafter(torch.tensor(0.5 / 128), torch.tensor(float("inf"))),
             torch.nextafter(torch.tensor(0.5 / 128), torch.tensor(float("-inf")))],
            device="cuda").view(3, 1)
        layer.factorized_linear.weight[:3, :1].zero_()
        for p in layer.group_factor_w:
            p.view(-1, layer.out_features, layer.in_features)[:, :3, :1].zero_()
    candidate = copy.deepcopy(layer)
    from model.modules.sharded_l1 import _compiled_l1_merge
    merge = lambda enabled: _compiled_l1_merge()(candidate, enabled)
    outputs = []
    for model, call in ((layer, layer.merged_parameters), (candidate, merge)):
        weight, bias = call(quantized)
        (weight.square().sum() + bias.square().sum()).backward()
        outputs.append((weight.detach(), bias.detach(),
                        [p.grad.detach().clone() for p in model.parameters()]))
    for left, right in zip(outputs[0][:2], outputs[1][:2]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for left, right in zip(outputs[0][2], outputs[1][2]):
        torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-6)


def test_compiled_clipping_model_copy_and_serialization(monkeypatch):
    monkeypatch.setattr(model_module, "get_feature_cls", lambda _: [_Feature])
    model = NNUE(_config()).model.cuda()
    model._compile_layer_clipping = True
    model.clip_weights(False)  # Warm the unbound compiled callable.
    snapshot = {name: p.detach().clone() for name, p in model.named_parameters()}
    copied = copy.deepcopy(model)
    with torch.no_grad():
        copied.layer_stacks.l1.linear.weight.fill_(5)
    copied.clip_weights(False)
    for name, p in model.named_parameters():
        torch.testing.assert_close(p, snapshot[name], rtol=0, atol=0)
    output = io.BytesIO()
    torch.save(copied, output)
    output.seek(0)
    restored = torch.load(output, weights_only=False)
    restored.clip_weights(False)
    for (name, p), (restored_name, q) in zip(copied.named_parameters(), restored.named_parameters()):
        assert name == restored_name
        torch.testing.assert_close(q, p, rtol=0, atol=0)
    def unexpected_compile():
        raise AssertionError("compiled clipping should use the eager fallback")
    monkeypatch.setattr(model_module, "_compiled_layer_clipping", unexpected_compile)
    with torch.autocast("cuda"):
        restored.clip_weights(False)
    # The enabled flag survives a move to CPU, where clipping falls back to eager.
    copied.cpu().clip_weights(False)


@pytest.mark.skipif(not os.environ.get("NNUE_TEST_REAL_REMAP_CACHE"),
                    reason="real stack-remap cache not provided")
def test_compiled_real_nettest_factors():
    from scripts.nettest_benchmark_settings import make_nettest_settings
    from model.modules.sharded_l1 import _compiled_l1_merge

    with open(os.environ["NNUE_TEST_REAL_REMAP_CACHE"], encoding="utf-8") as f:
        config, _, _ = make_nettest_settings(f.read().strip())
    torch.manual_seed(388)
    model = NNUE(config, sharding=(0, 4)).model.cuda()
    assert model.layer_stacks.count == 92
    with torch.no_grad():
        for p in model.layer_stacks.parameters():
            p.uniform_(-1.5, 1.5)
    candidate = copy.deepcopy(model)
    candidate._compile_layer_clipping = True
    model.clip_weights(False)
    candidate.clip_weights(False)
    for (name, p), (other_name, q) in zip(model.named_parameters(), candidate.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(q, p, rtol=0, atol=0)
    for quantized in (False, True):
        expected = model.layer_stacks.l1.merged_parameters(quantized)
        actual = _compiled_l1_merge()(candidate.layer_stacks.l1, quantized)
        for ref, result in zip(expected, actual):
            torch.testing.assert_close(result, ref, rtol=0, atol=0)

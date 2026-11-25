"""The compiled sharded L1 factor merge must match the eager version."""
import copy

import pytest
import torch

from model.modules.sharded_l1 import ShardedL1
from model.quantize import QuantizationConfig, QuantizationManager


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


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

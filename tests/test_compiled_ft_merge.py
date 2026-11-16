"""Compare experimental compiled FT materialization with eager autograd."""
import copy

import pytest
import torch

from model.modules import get_feature_cls
from model.modules.feature_transformer import ShardedFeatureTransformer
from model.quantize import QuantizationConfig, QuantizationManager


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("quantized", [False, True])
def test_compiled_ft_merge_matches_eager(quantized):
    torch.manual_seed(125)
    quant = QuantizationManager(QuantizationConfig())
    reference = ShardedFeatureTransformer(
        get_feature_cls("Full_Threats+PP_3Wide+K32H3^"), 1024, quant,
        rank=0, world_size=4,
    ).cuda()
    candidate = copy.deepcopy(reference)
    with torch.no_grad():
        for feature in reference.features:
            if hasattr(feature, "virtual_weight"):
                feature.virtual_weight.uniform_(-0.02, 0.02)
        candidate.load_state_dict(reference.state_dict())
    compiled = torch.compile(candidate.merged_weight_and_bias, fullgraph=True)
    expected_w, expected_b = reference.merged_weight_and_bias(quantized)
    actual_w, actual_b = compiled(quantized)
    torch.testing.assert_close(actual_w, expected_w, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(actual_b, expected_b, atol=1e-7, rtol=1e-6)
    grad_w = torch.randn_like(expected_w)
    grad_b = torch.randn_like(expected_b)
    expected = torch.autograd.grad((expected_w, expected_b), tuple(reference.parameters()),
                                   (grad_w, grad_b))
    actual = torch.autograd.grad((actual_w, actual_b), tuple(candidate.parameters()),
                                 (grad_w, grad_b))
    for lhs, rhs in zip(actual, expected):
        torch.testing.assert_close(lhs, rhs, atol=1e-4, rtol=1e-5)

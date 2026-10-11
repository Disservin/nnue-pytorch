"""Aggregation dispatch must retain fallbacks and permit A100 comparisons."""

import os
import sys
from functools import partial

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.modules.feature_transformer import fused_ft_functions as fused
from model.modules.feature_transformer.double_ft_functions import (
    double_feature_transform,
)


@pytest.mark.parametrize(
    "capability,batch,mode,expected",
    [
        ((8, 9), 4096, "auto", True),
        ((8, 9), 4095, "auto", False),
        ((9, 0), 1024, "auto", True),
        ((9, 0), 1023, "auto", False),
        ((8, 0), 131072, "auto", False),
        ((8, 0), 131072, "aggregated", True),
        ((8, 9), 131072, "scatter", False),
        ((9, 0), 131072, "scatter", False),
        ((7, 5), 131072, "aggregated", False),
    ],
)
def test_backward_dispatch(monkeypatch, capability, batch, mode, expected):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    assert fused._use_aggregated_backward(1024, batch, 288, torch.device("cuda:0"), mode=mode) == expected


@pytest.mark.parametrize("width,active", [(96, 33), (1154, 33), (8192, 33), (1024, 0), (1024, 289)])
def test_unsupported_shapes_keep_scatter(width, active):
    assert not fused._use_aggregated_backward(
        width, 131072, active, torch.device("cuda:0"), mode="aggregated"
    )


def test_hip_keeps_scatter(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "test")
    assert not fused._use_aggregated_backward(
        1024, 131072, 288, torch.device("cuda:0"), mode="aggregated"
    )


def test_invalid_mode():
    with pytest.raises(ValueError, match="mode must be"):
        fused._use_aggregated_backward(1024, 131072, 288, torch.device("cuda:0"), mode="typo")


@pytest.mark.skipif(
    not torch.cuda.is_available() or not fused._HAS_CUPY_KERNELS,
    reason="CUDA and CuPy required",
)
@pytest.mark.parametrize("width", [512, 1024, 1152, 4096])
def test_dispatched_aggregation_gradient_parity(monkeypatch, width):
    if torch.version.hip is not None or torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("NVIDIA Ampere or newer required")
    monkeypatch.setattr(
        fused,
        "_use_aggregated_backward",
        partial(fused._use_aggregated_backward, mode="aggregated"),
    )
    torch.manual_seed(274)
    batch, active, inputs = 4101, 33, 128
    # IDs overlap across rows but remain unique within each perspective.
    columns = torch.arange(active, device="cuda", dtype=torch.int32)
    rows = torch.arange(batch, device="cuda", dtype=torch.int32).view(-1, 1)
    white = (rows + columns) % inputs
    black = (rows * 7 + columns) % inputs
    white[rows[:, 0] % 7 == 0, 5:] = -1
    black[rows[:, 0] % 11 == 0] = -1
    us = (rows % 2).float()
    them = 1 - us
    weight = (torch.randn(inputs, width, device="cuda") * 0.1).requires_grad_()
    bias = torch.zeros(width, device="cuda", requires_grad=True)
    upstream = torch.randn(batch, width, device="cuda") * 0.001
    calls = []
    aggregate = fused.aggregated_ft_backward

    def record(*args, **kwargs):
        calls.append(True)
        return aggregate(*args, **kwargs)

    monkeypatch.setattr(fused, "aggregated_ft_backward", record)
    output = double_feature_transform(us, them, white, black, weight, bias, 1.0, width, "fused")
    output.backward(upstream)
    actual_weight, actual_bias = weight.grad.clone(), bias.grad.clone()
    assert len(calls) == 1
    weight.grad = bias.grad = None
    reference = double_feature_transform(us, them, white, black, weight, bias, 1.0, width, "torch")
    reference.backward(upstream)
    torch.testing.assert_close(output, reference, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(actual_weight, weight.grad, rtol=1e-3, atol=1e-5)
    torch.testing.assert_close(actual_bias, bias.grad, rtol=1e-3, atol=1e-5)

"""Check that native feature ranges preserve fused FT outputs and gradients."""

import pytest
import torch

from data_loader import distributed_features as features
from model.modules.feature_transformer import fused_ft_functions
from model.modules.feature_transformer.double_ft_functions import (
    double_feature_transform,
)
from scripts.benchmark_sharded_adamw import pack_saved_batch


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None
    or not fused_ft_functions._HAS_CUPY_KERNELS
    or features.unpack_native_distributed_rows is None,
    reason="NVIDIA CUDA, CuPy and Triton required",
)
@pytest.mark.parametrize("aggregated", [False, True])
def test_native_ranges_match_dense_ft(monkeypatch, aggregated):
    if aggregated and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("SM80+ required for aggregated FT")
    monkeypatch.setattr(
        fused_ft_functions, "_use_aggregated_backward", lambda *args: aggregated
    )
    torch.manual_seed(912)
    device = "cuda"
    count, width, active = 23, 256, 13
    row = torch.arange(count, device=device).reshape(-1, 1)
    col = torch.arange(active, device=device)
    white = ((row + col) % 41).int().masked_fill(col >= row % (active + 1), -1)
    black = ((row * 2 + col) % 41).int().masked_fill(col >= (row + 4) % (active + 1), -1)
    us = (row % 2).float()
    stacks = row.flatten() % 4
    batch = (us, 1 - us, white, black, us, us, stacks)

    dimensions, payloads = [], []
    for start, stop in [(0, 5), (5, 5), (5, count)]:
        u, v, native, *_ = pack_saved_batch(tuple(t[start:stop] for t in batch))
        dimensions.append(tuple(native[k] for k in ("size", "width", "nw", "nb")))
        payloads.append(torch.cat((native["packed"], u.flatten().view(torch.int32),
                                   v.flatten().view(torch.int32))))
    stride = max(p.numel() for p in payloads)
    gathered = torch.zeros((len(payloads), stride), dtype=torch.int32, device=device)
    for dest, payload in zip(gathered, payloads):
        dest[:payload.numel()] = payload
    packed = features._unpack_native_gathered(gathered, dimensions, 2, direct_packed_ft=True)
    assert packed.offsets == (0, 5, 5, count)
    assert packed.packed_payload is gathered
    torch.testing.assert_close(packed.layer_stack_indices, stacks)
    weight = (torch.rand(41, width, device=device) * 0.02).requires_grad_()
    bias = torch.full((width,), 0.3, device=device, requires_grad=True)
    expected = double_feature_transform(
        us, 1 - us, white, black, weight, bias, 1., width, sharded_backward=True
    )
    actual = double_feature_transform(
        packed.us, packed.them, packed.white_indices, packed.black_indices, weight, bias,
        1., width, sharded_backward=True,
        packed_payload=packed.packed_payload, packed_feature_width=active,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.rand_like(expected) - 0.5
    expected_grads = torch.autograd.grad(expected, (weight, bias), upstream)
    actual_grads = torch.autograd.grad(actual, (weight, bias), upstream)
    for result, target in zip(actual_grads, expected_grads):
        torch.testing.assert_close(result, target, rtol=1e-4, atol=2e-6)

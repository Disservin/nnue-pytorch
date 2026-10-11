import os
import sys

import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.modules.feature_transformer.double_ft_functions import (
    double_feature_transform,
)
from model.modules.feature_transformer.fused_ft_functions import _HAS_CUPY_KERNELS


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _HAS_CUPY_KERNELS,
    reason="CUDA and CuPy required for custom kernel",
)
@pytest.mark.parametrize("l1", [32, 64, 96, 264, 1024, 1152, 1280, 2048, 4096])
@pytest.mark.parametrize("batch_size", [1, 9])
@pytest.mark.parametrize("max_active", [96, 288])
def test_fused_double_ft(l1, batch_size, max_active):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    num_inputs = 100
    output_size = l1

    us = torch.randn(batch_size, 1, dtype=torch.float32, device="cuda")
    them = torch.randn(batch_size, 1, dtype=torch.float32, device="cuda")

    # ensure non-negative indices and some padding (-1)
    white_indices = torch.randint(
        0, num_inputs, (batch_size, max_active), dtype=torch.int32, device="cuda"
    )
    white_indices[:, -2:] = -1
    black_indices = torch.randint(
        0, num_inputs, (batch_size, max_active), dtype=torch.int32, device="cuda"
    )
    black_indices[:, -2:] = -1
    # Empty rows, unequal perspective lengths and a partial backward tile.
    for row in range(batch_size):
        white_indices[row, (row * 17) % max_active :] = -1
        black_indices[row, (row * 23 + 1) % max_active :] = -1

    weight = torch.randn(
        num_inputs,
        output_size,
        dtype=torch.float32,
        device="cuda",
        requires_grad=True,
    )
    bias = torch.randn(
        output_size, dtype=torch.float32, device="cuda", requires_grad=True
    )

    # 1) Fused kernel
    l0_fused = double_feature_transform(
        us,
        them,
        white_indices,
        black_indices,
        weight,
        bias,
        127.0,  # max_ft_activation
        l1,  # l1_size
        "fused",
    )

    # Distinct upstream gradients catch column-routing errors in tiled kernels.
    grad_output = torch.randn_like(l0_fused)
    loss_fused = (l0_fused * grad_output).sum()
    loss_fused.backward()

    grad_weight_fused = weight.grad.clone()
    grad_bias_fused = bias.grad.clone()

    # 2) Fallback
    weight.grad.zero_()
    bias.grad.zero_()

    l0_fallback = double_feature_transform(
        us,
        them,
        white_indices,
        black_indices,
        weight,
        bias,
        127.0,  # max_ft_activation
        l1,  # l1_size
        "torch",
    )

    loss_fallback = (l0_fallback * grad_output).sum()
    loss_fallback.backward()

    # Compare
    torch.testing.assert_close(l0_fused, l0_fallback, atol=1e-3, rtol=1e-4)

    torch.testing.assert_close(grad_weight_fused, weight.grad, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(grad_bias_fused, bias.grad, atol=1e-4, rtol=1e-3)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _HAS_CUPY_KERNELS,
    reason="CUDA and CuPy required for custom kernel",
)
@pytest.mark.parametrize("backend,batch_size", [("fused", 33), ("sparse", 33), ("fused", 4101)])
def test_compiled_double_ft(backend, batch_size):
    """Compiled CUDA forward/backward must match the PyTorch reference."""
    torch.manual_seed(274)
    width, active, inputs = 512, 33, 128
    rows = torch.arange(batch_size, device="cuda", dtype=torch.int32).view(-1, 1)
    columns = torch.arange(active, device="cuda", dtype=torch.int32)
    white = (rows + columns) % inputs
    black = (rows * 7 + columns) % inputs
    white[rows[:, 0] % 7 == 0, 5:] = -1
    black[rows[:, 0] % 11 == 0] = -1
    us = (rows % 2).float()
    them = 1 - us
    weight = (torch.randn(inputs, width, device="cuda") * 0.1).requires_grad_()
    bias = torch.zeros(width, device="cuda", requires_grad=True)
    upstream = torch.randn(batch_size, width, device="cuda") * 0.001

    def transform(weight, bias):
        return double_feature_transform(
            us, them, white, black, weight, bias, 1.0, width, backend
        )

    reference = double_feature_transform(
        us, them, white, black, weight, bias, 1.0, width, "torch"
    )
    expected_grads = torch.autograd.grad(reference, (weight, bias), upstream)
    compiled = torch.compile(transform, fullgraph=True)
    # Exercise initial tracing and reuse of the cached kernel launchers.
    for _ in range(2):
        output = compiled(weight, bias)
        actual_grads = torch.autograd.grad(output, (weight, bias), upstream)
        torch.testing.assert_close(output, reference, rtol=1e-4, atol=1e-5)
        for actual, expected in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _HAS_CUPY_KERNELS,
    reason="CUDA and CuPy required for custom kernel",
)
@pytest.mark.parametrize("backend,batch_size", [("sparse", 33), ("fused", 33), ("fused", 4101)])
def test_ft_custom_operator_contract(backend, batch_size):
    """Validate schemas, fake metadata, autograd and compiled execution."""
    from model.modules.feature_transformer.fused_ft_functions import (
        _fused_double_ft,
        _fused_double_ft_backward,
    )
    from model.modules.feature_transformer.sparse_linear_functions import (
        _cuda_sparse_linear,
        _cuda_sparse_linear_backward,
    )

    torch.manual_seed(274)
    width, active, inputs = 512, 33, 128
    rows = torch.arange(batch_size, device="cuda", dtype=torch.int32).view(-1, 1)
    columns = torch.arange(active, device="cuda", dtype=torch.int32)
    white = (rows + columns) % inputs
    black = (rows * 7 + columns) % inputs
    white[rows[:, 0] % 7 == 0, 5:] = -1
    black[rows[:, 0] % 11 == 0] = -1
    us = (rows % 2).float()
    them = 1 - us
    weight = (torch.randn(inputs, width, device="cuda") * 0.1).requires_grad_()
    bias = torch.zeros(width, device="cuda", requires_grad=True)
    # Exercise the contiguous gradient copy inside the opaque backward operation.
    grad = (torch.randn(width, batch_size, device="cuda") * 0.001).t()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        if backend == "sparse":
            torch.library.opcheck(_cuda_sparse_linear, (white, weight, bias))
            torch.library.opcheck(_cuda_sparse_linear_backward, (white, weight.detach(), grad))
        else:
            args = (us, them, white, black, weight, bias, 1.0, width)
            # The backward uses atomic accumulation; use the same tolerances
            # as the gradient parity checks rather than requiring repeatability.
            torch.library.opcheck(_fused_double_ft, args, rtol=1e-3, atol=1e-5)
            _, clamped = _fused_double_ft(*args)
            torch.library.opcheck(
                _fused_double_ft_backward,
                (us, them, white, black, weight.detach(), bias.detach(), clamped, grad, 1.0, width),
                # Atomic accumulation order varies between executions, as in
                # the eager-versus-reference gradient parity tests above.
                rtol=1e-3, atol=1e-5,
            )
    stream.synchronize()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _HAS_CUPY_KERNELS,
    reason="CUDA and CuPy required for custom kernel",
)
def test_fp16_weight_stream_matches_fp32_bitwise():
    """Reading the fake-quantized FT weight as fp16 must not change any value.

    k/256 weights inside the 127/256 clip range are exact in fp16, so the fp16
    launch has to reproduce the fp32 launch bit for bit; the only difference is
    that it reads half the bytes from the weight matrix.
    """
    from model.modules.feature_transformer import fused_ft_kernel as kernels
    from model.modules.feature_transformer.fused_ft_functions import _fused_double_ft

    l1, active, inputs, batch = 1024, 288, 4096, 37
    if not kernels.forward_reads_fp16_weights(active, l1):
        pytest.skip("fp16 weight stream needs the four-column launch geometry")

    torch.manual_seed(11)
    rows = torch.arange(batch, device="cuda", dtype=torch.int32).view(-1, 1)
    columns = torch.arange(active, device="cuda", dtype=torch.int32)
    white = (rows * 3 + columns * 7) % inputs
    black = (rows * 5 + columns * 11) % inputs
    white[rows[:, 0] % 5 == 0, 120:] = -1
    black[rows[:, 0] % 7 == 0, 200:] = -1
    us = torch.rand(batch, 1, device="cuda")
    them = 1 - us

    # Fake-quantized weights: integer steps of 1/256, clipped like the extractors do.
    scale = 256.0
    weight = torch.randint(-127, 128, (inputs, l1), device="cuda") / scale
    bias = torch.randint(-127, 128, (l1,), device="cuda") / scale

    kernels._fused_double_ft_forward_kernel_cache.clear()
    kernels._fused_double_ft_forward_fp16_kernel_cache.clear()

    # clamped activations are written straight from the weight sums, so they
    # pin the fp16 weight stream itself, not just the post-product output.
    l0_fp32, clamped_fp32 = _fused_double_ft(
        us, them, white, black, weight, bias, 1.0, l1, False
    )
    assert kernels._fused_double_ft_forward_fp16_kernel_cache == {}
    l0_fp16, clamped_fp16 = _fused_double_ft(
        us, them, white, black, weight, bias, 1.0, l1, True
    )
    assert kernels._fused_double_ft_forward_fp16_kernel_cache

    assert torch.equal(clamped_fp16, clamped_fp32)
    assert torch.equal(l0_fp16, l0_fp32)

    # The fp16 copy is internal to the op; autograd still sees the fp32 weight.
    narrow_weight = weight.clone().requires_grad_()
    out = double_feature_transform(
        us, them, white, black, narrow_weight, bias, 1.0, l1, "fused", fp16_weights=True
    )
    assert torch.equal(out, l0_fp32)
    grad = torch.autograd.grad(out.sum(), narrow_weight)[0]
    assert grad.shape == weight.shape and torch.isfinite(grad).all()


def test_fp16_weight_gate_requires_power_of_two_scale():
    from model.quantize import QuantizationConfig, QuantizationManager

    assert QuantizationManager(QuantizationConfig()).ft_weights_are_fp16_exact
    # A non power-of-two grid cannot land on the fp16 grid in general.
    assert not QuantizationManager(
        QuantizationConfig(ft_quantized_one=100.0)
    ).ft_weights_are_fp16_exact
    assert not QuantizationManager(
        QuantizationConfig(ft_quantized_one=1.5)
    ).ft_weights_are_fp16_exact

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

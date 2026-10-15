import os
import sys

import numpy as np
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


def _launch_forward(kernel, us, them, white, black, weight, bias, max_act, l1_size):
    batch = us.shape[0]
    l0 = torch.empty(batch, l1_size, dtype=torch.float32, device="cuda")
    clamped = torch.empty(
        batch, 4, l1_size // 2, dtype=torch.float32, device="cuda"
    )
    kernel(
        grid=(batch,),
        args=(
            us.data_ptr(),
            them.data_ptr(),
            white.data_ptr(),
            black.data_ptr(),
            weight.data_ptr(),
            bias.data_ptr(),
            np.float32(max_act),
            l0.data_ptr(),
            clamped.data_ptr(),
            np.int32(bias.shape[0]),
        ),
    )
    return l0, clamped


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _HAS_CUPY_KERNELS,
    reason="CUDA and CuPy required for custom kernel",
)
@pytest.mark.parametrize("l1", [512, 1024, 2048, 4096])
@pytest.mark.parametrize("max_active", [96, 288])
def test_four_column_forward_matches_scalar_launch(l1, max_active):
    """The four-column launch reproduces the scalar kernel bit for bit.

    Both cover the same columns in the same accumulation order, so any
    difference is a column-routing or clamp bug rather than rounding.
    """
    from model.modules.feature_transformer.fused_ft_kernel import (
        make_fused_double_ft_forward_strided_kernel,
        make_fused_double_ft_forward_vec4_kernel,
    )

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    num_inputs = 100
    batch_size = 37
    us = torch.randn(batch_size, 1, dtype=torch.float32, device="cuda")
    them = torch.randn(batch_size, 1, dtype=torch.float32, device="cuda")

    white = torch.randint(
        0, num_inputs, (batch_size, max_active), dtype=torch.int32, device="cuda"
    )
    black = torch.randint(
        0, num_inputs, (batch_size, max_active), dtype=torch.int32, device="cuda"
    )
    # Unequal perspective lengths, empty rows and padding.
    for row in range(batch_size):
        white[row, (row * 17) % max_active :] = -1
        black[row, (row * 13 + 5) % max_active :] = -1

    weight = torch.randn(num_inputs, l1, dtype=torch.float32, device="cuda")
    bias = torch.randn(l1, dtype=torch.float32, device="cuda")

    scalar = make_fused_double_ft_forward_strided_kernel(max_active, l1)
    vec4 = make_fused_double_ft_forward_vec4_kernel(max_active, l1)

    l0_scalar, clamped_scalar = _launch_forward(
        scalar, us, them, white, black, weight, bias, 0.99609375, l1
    )
    l0_vec4, clamped_vec4 = _launch_forward(
        vec4, us, them, white, black, weight, bias, 0.99609375, l1
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(l0_vec4, l0_scalar, rtol=0, atol=0)
    torch.testing.assert_close(clamped_vec4, clamped_scalar, rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _HAS_CUPY_KERNELS,
    reason="CUDA and CuPy required for custom kernel",
)
def test_forward_kernel_geometry_selection():
    """The four-column launch needs a whole number of columns per thread."""
    from model.modules.feature_transformer.fused_ft_kernel import (
        _forward_vec4_threads,
        make_fused_double_ft_forward_kernel,
        make_fused_double_ft_forward_strided_kernel,
        make_fused_double_ft_forward_vec4_kernel,
    )

    assert _forward_vec4_threads(512) == 128
    assert _forward_vec4_threads(1536) == 384
    # Too few threads to fill a block, and widths the launch cannot split.
    assert _forward_vec4_threads(128) is None
    assert _forward_vec4_threads(6) is None

    capability = torch.cuda.get_device_capability()
    expected = (
        make_fused_double_ft_forward_vec4_kernel
        if capability == (8, 0)
        else make_fused_double_ft_forward_strided_kernel
    )
    assert make_fused_double_ft_forward_kernel(288, 1024) is expected(288, 1024)


def _k_grid_weight(num_inputs, l1, k_max):
    """A fake-quantized FT weight: exactly the integers k / 256 it can hold."""
    k = torch.randint(
        -k_max, k_max + 1, (num_inputs, l1), dtype=torch.int64, device="cuda"
    )
    k[0, 0], k[0, 1], k[1, 0], k[1, 1] = k_max, -k_max, 1, -1
    return (k.to(torch.float32) / 256.0).contiguous()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _HAS_CUPY_KERNELS,
    reason="CUDA and CuPy required for custom kernel",
)
@pytest.mark.parametrize("l1", [512, 1024, 2048])
@pytest.mark.parametrize("stream_dtype", [torch.int8, torch.int16])
def test_narrow_weight_stream_matches_fp32_kernel(l1, stream_dtype):
    """The integer stream scales back to the fp32 grid value for value."""
    from model.modules.feature_transformer.fused_ft_kernel import (
        make_fused_double_ft_forward_vec4_kernel,
        narrow_ft_weight,
    )

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    k_max = torch.iinfo(stream_dtype).max
    num_inputs, max_active, batch_size = 100, 288, 37
    us = torch.randn(batch_size, 1, dtype=torch.float32, device="cuda")
    them = torch.randn(batch_size, 1, dtype=torch.float32, device="cuda")
    white = torch.randint(
        0, num_inputs, (batch_size, max_active), dtype=torch.int32, device="cuda"
    )
    black = torch.randint(
        0, num_inputs, (batch_size, max_active), dtype=torch.int32, device="cuda"
    )
    for row in range(batch_size):
        white[row, (row * 17) % max_active :] = -1
        black[row, (row * 13 + 5) % max_active :] = -1

    weight = _k_grid_weight(num_inputs, l1, k_max)
    bias = torch.randn(l1, dtype=torch.float32, device="cuda")
    narrow = narrow_ft_weight(weight, stream_dtype, 256.0)

    # The cast must land on the grid, including the extremes it can store.
    assert narrow[0, 0].item() == k_max
    assert narrow[0, 1].item() == -k_max
    assert torch.equal(narrow.float(), weight * 256.0)

    l0_wide, clamped_wide = _launch_forward(
        make_fused_double_ft_forward_vec4_kernel(max_active, l1),
        us,
        them,
        white,
        black,
        weight,
        bias,
        0.99609375,
        l1,
    )
    l0_narrow, clamped_narrow = _launch_forward(
        make_fused_double_ft_forward_vec4_kernel(max_active, l1, stream_dtype, 256.0),
        us,
        them,
        white,
        black,
        narrow,
        bias,
        0.99609375,
        l1,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(l0_narrow, l0_wide, rtol=0, atol=0)
    torch.testing.assert_close(clamped_narrow, clamped_wide, rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _HAS_CUPY_KERNELS,
    reason="CUDA and CuPy required for custom kernel",
)
def test_narrow_weight_stream_matches_fp32_forward():
    """Asking for the stream leaves the fused forward bit for bit unchanged."""
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    l1, max_active, batch_size, num_inputs = 1024, 288, 17, 128
    us = torch.randn(batch_size, 1, dtype=torch.float32, device="cuda")
    them = torch.randn(batch_size, 1, dtype=torch.float32, device="cuda")
    white = torch.randint(
        0, num_inputs, (batch_size, max_active), dtype=torch.int32, device="cuda"
    )
    black = torch.randint(
        0, num_inputs, (batch_size, max_active), dtype=torch.int32, device="cuda"
    )
    white[:, -3:] = -1
    black[:, -5:] = -1
    weight = _k_grid_weight(num_inputs, l1, 127)
    bias = torch.randn(l1, dtype=torch.float32, device="cuda")

    kwargs = (us, them, white, black, weight, bias, 0.99609375, l1)
    l0_wide = double_feature_transform(*kwargs, backend="fused")
    l0_narrow = double_feature_transform(
        *kwargs, backend="fused", weight_stream=(torch.int8, 256.0)
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(l0_narrow, l0_wide, rtol=0, atol=0)


def test_ft_weight_stream_dtype_gate():
    """Only feature sets whose clipped integers fit the stream are streamed."""
    from model.modules.features import get_feature_cls
    from model.quantize import QuantizationConfig, QuantizationManager

    manager = QuantizationManager(QuantizationConfig())
    assert manager.ft_weight_stream_dtype([torch.int8]) == torch.int8
    assert manager.ft_weight_stream_dtype([torch.int8, torch.int16]) == torch.int16
    assert manager.ft_weight_stream_dtype([torch.int8, torch.int32]) is None
    assert manager.ft_weight_stream_dtype([]) is None

    # The production feature set carries one int16-clipped extractor, so it
    # streams as int16 rather than int8.
    production = [
        cls(1024).EXPORT_WEIGHT_DTYPE
        for cls in get_feature_cls("Full_Threats+PP_3Wide+HalfKAv2_hm^")
    ]
    assert production == [torch.int8, torch.int8, torch.int16]
    assert manager.ft_weight_stream_dtype(production) == torch.int16

    # k / scale has to be exact in fp32 for the scale-back to be lossless.
    for scale in (100.0, 256.5, -256.0):
        other = QuantizationManager(QuantizationConfig(ft_quantized_one=scale))
        assert other.ft_weight_stream_dtype([torch.int8]) is None

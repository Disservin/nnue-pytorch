"""Pinned copies must outlive the native buffers and preserve tensor values."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader._native import _pin_and_move


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA pinned memory required")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("copy_mode", ["memcpy", "torch"])
@pytest.mark.parametrize("kind", ["floats", "ints", "strided", "conversion", "neg_view", "empty"])
def test_native_pinned_copy(device, copy_mode, kind):
    if kind == "floats":
        source = torch.tensor([0.0, -0.0, float("nan"), float("inf"), -1.25])
    elif kind == "ints":
        source = torch.tensor([-2147483648, -1, 0, 2147483647], dtype=torch.int32)
    elif kind == "strided":
        source = torch.arange(24, dtype=torch.int32).reshape(4, 6)[:, ::2]
    elif kind == "neg_view":
        source = torch.arange(24, dtype=torch.float32)._neg_view()
    elif kind == "empty":
        source = torch.empty(0, dtype=torch.int32)
    else:
        source = torch.arange(24, dtype=torch.int32)
    dtype = torch.int64 if kind == "conversion" else source.dtype
    expected = source.to(dtype=dtype).clone()
    result = _pin_and_move(source, device, use_pinned_memory=True, dtype=dtype, copy_mode=copy_mode)
    # The native loader may immediately recycle the original allocation.
    source.fill_(42)
    if device == "cpu" and result.numel():
        assert result.is_pinned()
    actual = result.cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    if kind == "floats":
        assert torch.equal(torch.signbit(actual), torch.signbit(expected))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_pinned_copy_on_another_stream():
    stream = torch.cuda.Stream()
    expected = torch.arange(131072, dtype=torch.int32)
    source = expected.clone()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(2_000_000)
        result = _pin_and_move(source, "cuda", use_pinned_memory=True)
        source.zero_()
        # Allocate another pinned copy while the first DMA is still pending.
        other = _pin_and_move(source, "cuda", use_pinned_memory=True)
    stream.synchronize()
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
    assert torch.count_nonzero(other).item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_invalid_pinned_copy_mode():
    with pytest.raises(ValueError, match="copy_mode must be"):
        _pin_and_move(torch.zeros(1), "cpu", use_pinned_memory=True, copy_mode="typo")

"""Safety, fallback and autograd tests for optional bucketed L1 bias."""
import subprocess
import sys

import pytest
import torch

import model.modules.bucketed_bias as bb


def _inputs(device, count=4, width=3, batch=5, dtype=torch.float32):
    values = torch.randn(batch, width, device=device, dtype=dtype, requires_grad=True)
    bias = torch.randn(count * width, device=device, dtype=dtype, requires_grad=True)
    indices = torch.arange(batch, device=device, dtype=torch.int64).remainder(count)
    return values, bias, indices


def _compare(values, bias, indices, count, width):
    actual = bb.bucketed_bias_add(values, bias, indices, count, width)
    expected = values + bias.view(count, width)[indices]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    grad = torch.randn_like(expected)
    actual_grads = torch.autograd.grad(actual, (values, bias), grad, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (values, bias), grad)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-3)
    return actual


def test_cpu_fallback_negative_indices_and_invalid_bounds():
    values, bias, indices = _inputs("cpu")
    indices[:] = torch.tensor([-4, -1, 0, 1, 3])
    actual = _compare(values, bias, indices, 4, 3)
    assert type(actual.grad_fn).__name__ != "_BucketedBiasAddBackward"
    for invalid in (-5, 4, -(2**63), 2**63 - 1):
        bad = indices.clone()
        bad[0] = invalid
        with pytest.raises(IndexError):
            bb.bucketed_bias_add(values, bias, bad, 4, 3)


def test_cpu_fallback_supports_broadcast_and_int32_indices():
    values, bias, indices = _inputs("cpu", batch=1)
    indices = torch.tensor([0, 1, 2], dtype=torch.int32)
    actual = _compare(values, bias, indices, 4, 3)
    assert actual.shape == (3, 3)


def _gpu_fast_available():
    return (torch.cuda.is_available() and torch.version.hip is None and bb.cp is not None
            and bb.triton is not None and bb._bucketed_bias_gradient is not None
            and torch.cuda.get_device_capability()[0] >= 8)


gpu = pytest.mark.skipif(not _gpu_fast_available(), reason="NVIDIA SM80+, CuPy and Triton required")


@gpu
def test_gpu_fast_path_negative_and_noncontiguous_indices():
    values, bias, _ = _inputs("cuda", count=92, width=32, batch=7)
    storage = torch.tensor([-92, 17, -1, 17, 0, 17, 91, 17, -22, 17, 4, 17, 20, 17],
                           dtype=torch.int64, device="cuda")
    indices = storage[::2]
    assert not indices.is_contiguous()
    actual = _compare(values, bias, indices, 92, 32)
    assert type(actual.grad_fn).__name__ == "_BucketedBiasAddBackward"


@gpu
def test_gpu_router_masks_invalid_and_large_int64_values():
    ids = torch.tensor([-(2**63), -93, -92, -1, 0, 91, 92, 2**63 - 1],
                       dtype=torch.int64, device="cuda")
    rows, counts = bb._route_bias_rows(ids, 92)
    expected = torch.zeros(92, dtype=torch.int32, device="cuda")
    expected[0] = 2
    expected[91] = 2
    torch.testing.assert_close(counts, expected, rtol=0, atol=0)
    assert sorted(rows[0, :2].cpu().tolist()) == [2, 4]
    assert sorted(rows[91, :2].cpu().tolist()) == [3, 5]


@gpu
@pytest.mark.parametrize("invalid", [-93, 92, -(2**63), 2**63 - 1])
def test_gpu_invalid_indices_preserve_pytorch_error(invalid):
    # CUDA's device-side index assertion poisons its context. Keep each case
    # in a child process so it cannot affect the rest of the test suite.
    program = r'''
import sys
import torch
from model.modules.bucketed_bias import bucketed_bias_add
values = torch.randn(1, 32, device="cuda", requires_grad=True)
bias = torch.randn(92 * 32, device="cuda", requires_grad=True)
indices = torch.tensor([int(sys.argv[1])], device="cuda", dtype=torch.int64)
try:
    bucketed_bias_add(values, bias, indices, 92, 32).sum().backward()
    torch.cuda.synchronize()
except (RuntimeError, IndexError):
    sys.exit(0)
sys.exit(3)
'''
    result = subprocess.run([sys.executable, "-c", program, str(invalid)],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr


@gpu
def test_gpu_scratch_limit_boundary(monkeypatch):
    values, bias, indices = _inputs("cuda", count=4, width=3, batch=5)
    required = bb._scratch_bytes(5, 4, 3)
    monkeypatch.setattr(bb, "_MAX_SCRATCH_BYTES", required)
    assert bb._eligible(values, bias, indices, 4, 3)
    monkeypatch.setattr(bb, "_MAX_SCRATCH_BYTES", required - 1)
    assert not bb._eligible(values, bias, indices, 4, 3)


@gpu
@pytest.mark.parametrize("reason", ["scratch", "deterministic", "dtype", "empty", "broadcast", "index_dtype", "no_bias_grad"])
def test_gpu_unsupported_cases_use_eager(reason, monkeypatch):
    values, bias, indices = _inputs("cuda")
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        if reason == "scratch":
            monkeypatch.setattr(bb, "_MAX_SCRATCH_BYTES", 0)
        elif reason == "deterministic":
            torch.use_deterministic_algorithms(True)
        elif reason == "dtype":
            values, bias = values.double(), bias.double()
        elif reason == "empty":
            values, indices = values[:0], indices[:0]
        elif reason == "broadcast":
            values = values[:1]
        elif reason == "index_dtype":
            indices = indices.int()
        elif reason == "no_bias_grad":
            bias = bias.detach()
        assert not bb._eligible(values, bias, indices, 4, 3)
        actual = bb.bucketed_bias_add(values, bias, indices, 4, 3)
        expected = values + bias.view(4, 3)[indices]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert type(actual.grad_fn).__name__ != "_BucketedBiasAddBackward"
    finally:
        torch.use_deterministic_algorithms(old_deterministic)


@gpu
def test_gpu_higher_order_gradients_match_eager():
    values, bias, indices = _inputs("cuda")
    indices[0] = -1
    actual = bb.bucketed_bias_add(values, bias, indices, 4, 3)
    assert type(actual.grad_fn).__name__ == "_BucketedBiasAddBackward"
    expected = values + bias.view(4, 3)[indices]
    actual_first = torch.autograd.grad(actual.square().sum(), (values, bias), create_graph=True)
    expected_first = torch.autograd.grad(expected.square().sum(), (values, bias), create_graph=True)
    actual_second = torch.autograd.grad(sum(g.square().sum() for g in actual_first), (values, bias))
    expected_second = torch.autograd.grad(sum(g.square().sum() for g in expected_first), (values, bias))
    for got, ref in zip(actual_first + actual_second, expected_first + expected_second):
        torch.testing.assert_close(got, ref, rtol=1e-6, atol=1e-6)


@gpu
@pytest.mark.parametrize("count,width", [(257, 3), (4, 129)])
def test_gpu_shape_limits_fall_back(count, width):
    values, bias, indices = _inputs("cuda", count=count, width=width)
    assert not bb._eligible(values, bias, indices, count, width)
    actual = _compare(values, bias, indices, count, width)
    assert type(actual.grad_fn).__name__ != "_BucketedBiasAddBackward"


@gpu
def test_gpu_nonfinite_gradient_propagation():
    values, bias, indices = _inputs("cuda", count=4, width=3, batch=5)
    indices[:] = torch.tensor([0, 0, 1, 1, 3], device="cuda")
    grad = torch.ones_like(values)
    grad[0, 0], grad[1, 0], grad[2, 1] = float("inf"), float("-inf"), float("nan")
    actual = bb.bucketed_bias_add(values, bias, indices, 4, 3)
    expected = values + bias.view(4, 3)[indices]
    actual_grads = torch.autograd.grad(actual, (values, bias), grad, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (values, bias), grad)
    for got, ref in zip(actual_grads, expected_grads):
        torch.testing.assert_close(got, ref, rtol=0, atol=0, equal_nan=True)


@gpu
def test_gpu_noncontiguous_bias_falls_back():
    values, _, indices = _inputs("cuda")
    bias = torch.randn(24, device="cuda")[::2].requires_grad_()
    assert not bias.is_contiguous()
    assert not bb._eligible(values, bias, indices, 4, 3)
    _compare(values, bias, indices, 4, 3)


@gpu
def test_gpu_nondefault_stream():
    values, bias, indices = _inputs("cuda", count=92, width=32, batch=1025)
    stream = torch.cuda.Stream(device=values.device)
    stream.wait_stream(torch.cuda.current_stream(values.device))
    with torch.cuda.stream(stream):
        actual = _compare(values, bias, indices, 92, 32)
        assert type(actual.grad_fn).__name__ == "_BucketedBiasAddBackward"
    stream.synchronize()


@gpu
def test_gpu_autocast_and_missing_dependencies_fall_back(monkeypatch):
    values, bias, indices = _inputs("cuda")
    with torch.autocast("cuda"):
        assert not bb._eligible(values, bias, indices, 4, 3)
        actual = bb.bucketed_bias_add(values, bias, indices, 4, 3)
        assert type(actual.grad_fn).__name__ != "_BucketedBiasAddBackward"
    monkeypatch.setattr(bb, "cp", None)
    assert not bb._eligible(values, bias, indices, 4, 3)
    _compare(values, bias, indices, 4, 3)


@gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices required")
def test_gpu_tensor_device_differs_from_current_device():
    previous = torch.cuda.current_device()
    try:
        torch.cuda.set_device(0)
        values, bias, indices = _inputs("cuda:1", count=92, width=32, batch=1025)
        actual = _compare(values, bias, indices, 92, 32)
        assert type(actual.grad_fn).__name__ == "_BucketedBiasAddBackward"
        assert torch.cuda.current_device() == 0
    finally:
        torch.cuda.set_device(previous)


@gpu
def test_gpu_cuda_graph_capture_falls_back():
    values, bias, indices = _inputs("cuda", count=92, width=32, batch=128)
    stream = torch.cuda.Stream(device=values.device)
    stream.wait_stream(torch.cuda.current_stream(values.device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            values + bias.view(92, 32)[indices]
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = bb.bucketed_bias_add(values, bias, indices, 92, 32)
    assert type(actual.grad_fn).__name__ != "_BucketedBiasAddBackward"
    graph.replay()
    torch.cuda.synchronize(values.device)
    torch.testing.assert_close(actual, values + bias.view(92, 32)[indices], rtol=0, atol=0)

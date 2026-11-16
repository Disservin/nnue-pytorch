"""Check experimental bucketed bias-add gradients against eager PyTorch."""
import pytest
import torch

try:
    from scripts.experimental_bias_add import bucketed_bias_add
except (ImportError, OSError, RuntimeError):
    bucketed_bias_add = None


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None
                    or bucketed_bias_add is None,
                    reason="CUDA, CuPy and Triton are required")
@pytest.mark.parametrize("batch", [0, 1, 3073, 32768])
@pytest.mark.parametrize("count", [92, 192])
def test_bucketed_bias_add(batch, count):
    torch.manual_seed(182)
    width = 32
    values = torch.randn(batch, width, device="cuda", requires_grad=True)
    bias = torch.randn(count * width, device="cuda", requires_grad=True)
    # Include both a very common and sparsely populated bucket.
    indices = torch.randint(count, (batch,), dtype=torch.int64, device="cuda")
    indices[::2] = 0
    grad = torch.randn_like(values)
    expected = values + bias.view(count, width)[indices]
    actual = bucketed_bias_add(values, bias, indices, count, width)
    expected_grads = torch.autograd.grad(expected, (values, bias), grad, retain_graph=True)
    actual_grads = torch.autograd.grad(actual, (values, bias), grad)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-3)

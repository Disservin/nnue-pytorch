"""Bucketed reduction for sharded L1 bias gradients, with an eager fallback."""
from functools import lru_cache

import torch

try:
    import cupy as cp
    import triton
    from .grouped_linear import _bucketed_bias_gradient
except (ImportError, OSError, RuntimeError):
    cp = None
    triton = None
    _bucketed_bias_gradient = None


_SPLIT = 8
_MAX_SCRATCH_BYTES = 64 * 1024 * 1024
_ROUTE_KERNELS = {}


def _scratch_bytes(batch, count, width):
    # Conservatively include route rows/counts, partials and result, plus
    # possible contiguous copies of indices and the incoming gradient.
    return (4 * count * batch + 4 * count + 4 * count * _SPLIT * width
            + 4 * count * width + 8 * batch + 4 * batch * width)


def _eager_bias_add(values, bias, indices, count, out_features):
    return values + bias.view(count, out_features)[indices]


@lru_cache(maxsize=None)
def _supported_cuda_device(device):
    return torch.cuda.get_device_capability(device)[0] >= 8


def _eligible(values, bias, indices, count, out_features):
    """Check only host-visible metadata; never synchronize to inspect IDs."""
    if (cp is None or triton is None or _bucketed_bias_gradient is None
            or torch.version.hip is not None or not torch.is_grad_enabled()
            or torch.are_deterministic_algorithms_enabled() or torch.is_autocast_enabled()
            or not isinstance(count, int) or not isinstance(out_features, int)
            or not 1 <= count <= 256 or not 1 <= out_features <= 128
            or values.device.type != "cuda" or values.dtype != torch.float32
            or bias.device != values.device or bias.dtype != torch.float32
            or indices.device != values.device or indices.dtype != torch.int64
            or values.ndim != 2 or bias.ndim != 1 or indices.ndim != 1
            or not bias.is_contiguous() or not bias.requires_grad
            or values.shape != (indices.numel(), out_features)
            or bias.numel() != count * out_features or indices.numel() == 0):
        return False
    batch = indices.numel()
    scratch = _scratch_bytes(batch, count, out_features)
    if not (batch <= 2**31 - 1 and count * batch <= 2**31 - 1
            and scratch <= _MAX_SCRATCH_BYTES and _supported_cuda_device(values.device)):
        return False
    # Raw-kernel graph replay has not been validated; keep graph capture on
    # the ordinary PyTorch path without synchronizing the device.
    with torch.cuda.device(values.device):
        return not torch.cuda.is_current_stream_capturing()


def _route_bias_rows(indices, count):
    """Route valid int64 IDs, normalize negative IDs, and mask invalid writes.

    The eager forward gather performs PyTorch's usual bounds validation.
    This separate guard also protects all router memory accesses without a
    device-to-host synchronization, including for very large int64 values.
    """
    kernel = _ROUTE_KERNELS.get(count)
    if kernel is None:
        kernel = cp.RawKernel(
            f"#define BUCKETS {count}\n" + r"""
extern "C" __global__ void route_bias(
    const long long* ids, int* rows, int* counts, int batch) {
    __shared__ int used[BUCKETS], base[BUCKETS];
    int t = threadIdx.x, r = blockIdx.x * blockDim.x + t;
    if (t < BUCKETS) used[t] = 0;
    __syncthreads();
    int b = 0, pos = 0, valid = 0;
    if (r < batch) {
        long long raw = ids[r];
        if (raw < 0) raw += BUCKETS;
        if (raw >= 0 && raw < BUCKETS) {
            b = (int)raw;
            pos = atomicAdd(&used[b], 1);
            valid = 1;
        }
    }
    __syncthreads();
    if (t < BUCKETS) base[t] = atomicAdd(&counts[t], used[t]);
    __syncthreads();
    if (valid) rows[(long long)b * batch + base[b] + pos] = r;
}
""", "route_bias")
        _ROUTE_KERNELS[count] = kernel
    batch = indices.numel()
    rows = torch.empty((count, batch), dtype=torch.int32, device=indices.device)
    counts = torch.zeros(count, dtype=torch.int32, device=indices.device)
    if batch:
        with cp.cuda.Device(indices.device.index):
            stream = cp.cuda.ExternalStream(torch.cuda.current_stream(indices.device).cuda_stream)
            kernel((triton.cdiv(batch, 256),), (256,),
                   (indices.data_ptr(), rows.data_ptr(), counts.data_ptr(), batch), stream=stream)
    return rows, counts


class _BucketedBiasAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, bias, indices, count, out_features):
        ctx.save_for_backward(indices)
        ctx.count, ctx.out_features = count, out_features
        # Keep the same forward operation and bounds checks as eager PyTorch.
        return _eager_bias_add(values, bias, indices, count, out_features)

    @staticmethod
    def backward(ctx, grad):
        (indices,) = ctx.saved_tensors
        count, width = ctx.count, ctx.out_features
        if torch.is_grad_enabled():
            # Preserve higher-order autograd through the reduction.
            bias_grad = grad.new_zeros((count, width)).index_add(
                0, indices.remainder(count), grad
            )
            return grad, bias_grad.reshape(-1), None, None, None
        with torch.cuda.device(grad.device):
            indices, grad = indices.contiguous(), grad.contiguous()
            rows, counts = _route_bias_rows(indices, count)
            partial = torch.empty((count, _SPLIT, width), dtype=grad.dtype, device=grad.device)
            _bucketed_bias_gradient[(count, _SPLIT)](
                grad, rows, counts, partial, len(indices), width, _SPLIT,
                max(16, triton.next_power_of_2(width)), num_warps=4,
            )
            bias_grad = partial.sum(dim=1).reshape(-1)
        return grad, bias_grad, None, None, None


def bucketed_bias_add(values, bias, indices, count, out_features):
    """Use the guarded fast reduction when supported, otherwise eager PyTorch."""
    if _eligible(values, bias, indices, count, out_features):
        return _BucketedBiasAdd.apply(values, bias, indices, count, out_features)
    return _eager_bias_add(values, bias, indices, count, out_features)

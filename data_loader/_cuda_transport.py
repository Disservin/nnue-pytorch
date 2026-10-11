"""Restore padded feature indices after transferring only active native IDs."""

from functools import cache

import cupy as cp
import numpy as np
import torch


@cache
def _unpack_kernel():
    return cp.RawKernel(
        r"""
extern "C" __global__ void unpack_sparse_indices(
    const int* packed, int* output, int batch, int active, int nw, int nb
) {
    const int row = blockIdx.x;
    const int* white_offsets = packed;
    const int* black_offsets = white_offsets + batch + 1;
    const int* white_ids = black_offsets + batch + 1;
    const int* black_ids = white_ids + nw;
    const int* piece_counts = black_ids + nb;
    const int white_start = white_offsets[row], black_start = black_offsets[row];
    const int white_count = white_offsets[row + 1] - white_start;
    const int black_count = black_offsets[row + 1] - black_start;
    for (int k = threadIdx.x; k < active; k += blockDim.x) {
        output[(size_t)row * active + k] = k < white_count ? white_ids[white_start + k] : -1;
        output[(size_t)(batch + row) * active + k] = k < black_count ? black_ids[black_start + k] : -1;
    }
    if (threadIdx.x == 0)
        output[(size_t)2 * batch * active + row] = piece_counts[row];
}
""",
        "unpack_sparse_indices",
    )


@torch.compiler.disable
def unpack_sparse_indices(packed, batch_size, max_active, num_white, num_black):
    """Restore [white rows, black rows, piece counts] on the current CUDA stream."""
    output = torch.empty(
        batch_size * (2 * max_active + 1), dtype=torch.int32, device=packed.device
    )
    if batch_size:
        with cp.cuda.Device(packed.device.index):
            stream = cp.cuda.ExternalStream(torch.cuda.current_stream(packed.device).cuda_stream)
            _unpack_kernel()(
                (batch_size,),
                (128,),
                (
                    packed.data_ptr(), output.data_ptr(), np.int32(batch_size),
                    np.int32(max_active), np.int32(num_white), np.int32(num_black),
                ),
                stream=stream,
            )
    return output

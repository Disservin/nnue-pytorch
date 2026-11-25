"""Put feature IDs back into padded rows on the GPU."""

import torch
import triton
import triton.language as tl


@triton.jit
def _unpack_counted_rows_kernel(counts, ends, ids, output,
                                WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    count = tl.load(counts + row)
    start = tl.load(ends + row) - count
    col = tl.arange(0, BLOCK)
    values = tl.load(ids + start + col, (col < WIDTH) & (col < count), other=-1)
    tl.store(output + row.to(tl.int64) * WIDTH + col, values, col < WIDTH)


@torch.compiler.disable
def unpack_counted_rows(counts: torch.Tensor, ids: torch.Tensor, width: int):
    """Expand compact IDs with per-row counts on the current CUDA stream."""
    rows = counts.numel()
    result = torch.empty((rows, width), dtype=torch.int32, device=counts.device)
    if rows and width:
        ends = torch.cumsum(counts, dim=0, dtype=torch.int64)
        with torch.cuda.device(counts.device):
            _unpack_counted_rows_kernel[(rows,)](
                counts, ends, ids, result, width, BLOCK=triton.next_power_of_2(width)
            )
    return result


@triton.jit
def _unpack_kernel(
    packed: tl.tensor,
    output: tl.tensor,
    batch: tl.tensor,
    active: tl.constexpr,
    nw: tl.tensor,
    nb: tl.tensor,
    BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    perspective = tl.program_id(1)
    offsets = perspective * (batch + 1)
    start = tl.load(packed + offsets + row)
    end = tl.load(packed + offsets + row + 1)
    ids = 2 * (batch + 1) + tl.where(perspective == 0, 0, nw)
    k = tl.arange(0, BLOCK)
    values = tl.load(
        packed + ids + start + k, (k < active) & (k < end - start), other=-1
    )
    destination = (perspective * batch + row).to(tl.int64) * active
    tl.store(output + destination + k, values, k < active)
    # The white perspective also copies the layer-stack index.
    if perspective == 0:
        value = tl.load(packed + 2 * (batch + 1) + nw + nb + row)
        tl.store(output + 2 * batch.to(tl.int64) * active + row, value)


@torch.compiler.disable
def unpack_sparse_indices(
    packed: torch.Tensor,
    batch_size: int,
    max_active: int,
    num_white: int,
    num_black: int,
) -> torch.Tensor:
    """Restore padded rows and copy layer-stack indices on the current CUDA stream."""
    if packed.dtype != torch.int32 or packed.ndim != 1 or not packed.is_contiguous():
        raise ValueError("packed must be a contiguous one-dimensional int32 tensor")
    if min(batch_size, max_active, num_white, num_black) < 0:
        raise ValueError("transport dimensions must be nonnegative")
    if packed.numel() != 3 * batch_size + 2 + num_white + num_black:
        raise ValueError("packed size does not match transport dimensions")
    output = torch.empty(
        batch_size * (2 * max_active + 1), dtype=torch.int32, device=packed.device
    )
    if batch_size:
        with torch.cuda.device(packed.device):
            _unpack_kernel[(batch_size, 2)](
                packed,
                output,
                batch_size,
                max_active,
                num_white,
                num_black,
                BLOCK=triton.next_power_of_2(max(1, max_active)),
            )
    return output

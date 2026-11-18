"""Put feature IDs back into padded rows on the GPU."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["STRIDE"])
def _unpack_native_distributed_kernel(payload, metadata, white, black, us, them, stacks,
                                      STRIDE, WIDTH: tl.constexpr,
                                      BLOCK: tl.constexpr, FEATURE_RANGES: tl.constexpr):
    row, rank, perspective = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n = tl.load(metadata + 4 * rank)
    nw = tl.load(metadata + 4 * rank + 1)
    nb = tl.load(metadata + 4 * rank + 2)
    offset = tl.load(metadata + 4 * rank + 3)
    valid = row < n
    src = payload + rank.to(tl.int64) * STRIDE
    offs = src + perspective * (n + 1) + row
    start = tl.load(offs, mask=valid, other=0)
    end = tl.load(offs + 1, mask=valid, other=0)
    global_row = offset + row
    if FEATURE_RANGES:
        base = rank.to(tl.int64) * STRIDE + 2 * (n + 1) + perspective * nw
        if perspective == 0:
            tl.store(white + 2 * global_row, base + start, mask=valid)
            tl.store(white + 2 * global_row + 1, base + end, mask=valid)
        else:
            tl.store(black + 2 * global_row, base + start, mask=valid)
            tl.store(black + 2 * global_row + 1, base + end, mask=valid)
    else:
        col = tl.arange(0, BLOCK)
        value = tl.load(src + 2 * (n + 1) + perspective * nw + start + col,
                        mask=valid & (col < WIDTH) & (col < end - start), other=-1)
        if perspective == 0:
            tl.store(white + global_row.to(tl.int64) * WIDTH + col, value,
                     mask=valid & (col < WIDTH))
        else:
            tl.store(black + global_row.to(tl.int64) * WIDTH + col, value,
                     mask=valid & (col < WIDTH))
    if perspective == 0:
        stack_base = 2 * (n + 1) + nw + nb
        u = tl.load(src + stack_base + n + row, mask=valid, other=0).to(tl.float32, bitcast=True)
        v = tl.load(src + stack_base + 2 * n + row, mask=valid, other=0).to(tl.float32, bitcast=True)
        s = tl.load(src + stack_base + row, mask=valid, other=0)
        tl.store(us + global_row, u, mask=valid)
        tl.store(them + global_row, v, mask=valid)
        tl.store(stacks + global_row, s, mask=valid)


@torch.compiler.disable
def unpack_native_distributed_rows(payload, dimensions, width, *, feature_ranges=False):
    total = sum(n for n, _, _, _ in dimensions)
    device = payload.device
    white = torch.empty((total, 2 if feature_ranges else width), dtype=torch.int32, device=device)
    black = torch.empty_like(white)
    us = torch.empty((total, 1), dtype=torch.float32, device=device)
    them = torch.empty_like(us)
    stacks = torch.empty(total, dtype=torch.int64, device=device)
    if total:
        meta, offset = [], 0
        for n, _, nw, nb in dimensions:
            meta.append([n, nw, nb, offset])
            offset += n
        metadata = torch.tensor(meta, dtype=torch.int64, device=device)
        with torch.cuda.device(device):
            _unpack_native_distributed_kernel[(max(d[0] for d in dimensions), len(dimensions), 2)](
                payload, metadata, white, black, us, them, stacks,
                payload.shape[1], width, BLOCK=triton.next_power_of_2(max(1, width)),
                FEATURE_RANGES=feature_ranges,
            )
    return us, them, white, black, stacks


@triton.jit(do_not_specialize=["STRIDE", "TOTAL"])
def _unpack_distributed_kernel(payload, metadata, ends, white, black, us, them, stacks,
                               STRIDE, TOTAL, WIDTH: tl.constexpr,
                               BLOCK: tl.constexpr):
    row = tl.program_id(0)
    rank = tl.program_id(1)
    perspective = tl.program_id(2)
    n = tl.load(metadata + 5 * rank)
    nw = tl.load(metadata + 5 * rank + 1)
    offset = tl.load(metadata + 5 * rank + 2)
    base_white = tl.load(metadata + 5 * rank + 3)
    base_black = tl.load(metadata + 5 * rank + 4)
    valid = row < n
    src = payload + rank.to(tl.int64) * STRIDE
    count = tl.load(src + perspective * n + row, mask=valid, other=0)
    global_row = offset + row
    end = tl.load(ends + perspective * TOTAL + global_row, mask=valid, other=0)
    base = tl.where(perspective == 0, base_white, base_black)
    start = end - count - base
    col = tl.arange(0, BLOCK)
    value = tl.load(src + 5 * n + perspective * nw + start + col,
                    mask=valid & (col < WIDTH) & (col < count), other=-1)
    if perspective == 0:
        tl.store(white + global_row.to(tl.int64) * WIDTH + col, value,
                 mask=valid & (col < WIDTH))
        u = tl.load(src + 2 * n + row, mask=valid, other=0).to(tl.float32, bitcast=True)
        v = tl.load(src + 3 * n + row, mask=valid, other=0).to(tl.float32, bitcast=True)
        s = tl.load(src + 4 * n + row, mask=valid, other=0)
        tl.store(us + global_row, u, mask=valid)
        tl.store(them + global_row, v, mask=valid)
        tl.store(stacks + global_row, s, mask=valid)
    else:
        tl.store(black + global_row.to(tl.int64) * WIDTH + col, value,
                 mask=valid & (col < WIDTH))


@torch.compiler.disable
def unpack_distributed_rows(payload, dimensions, width):
    """Reconstruct all gathered rank payloads directly in global-batch order."""
    total = sum(n for n, _, _, _ in dimensions)
    device = payload.device
    white = torch.empty((total, width), dtype=torch.int32, device=device)
    black = torch.empty_like(white)
    us = torch.empty((total, 1), dtype=torch.float32, device=device)
    them = torch.empty_like(us)
    stacks = torch.empty(total, dtype=torch.int64, device=device)
    if not total:
        return us, them, white, black, stacks
    segments = [payload[rank, :n] for rank, (n, _, _, _) in enumerate(dimensions)]
    segments += [payload[rank, n:2*n] for rank, (n, _, _, _) in enumerate(dimensions)]
    ends = torch.cumsum(torch.cat(segments), dim=0, dtype=torch.int64)
    meta = []
    offset = base_white = 0
    base_black = sum(nw for _, _, nw, _ in dimensions)
    for n, _, nw, nb in dimensions:
        meta.append([n, nw, offset, base_white, base_black])
        offset += n
        base_white += nw
        base_black += nb
    metadata = torch.tensor(meta, dtype=torch.int64, device=device)
    with torch.cuda.device(device):
        _unpack_distributed_kernel[(max(d[0] for d in dimensions), len(dimensions), 2)](
            payload, metadata, ends, white, black, us, them, stacks,
            payload.shape[1], total, width, BLOCK=triton.next_power_of_2(max(1, width)),
        )
    return us, them, white, black, stacks


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

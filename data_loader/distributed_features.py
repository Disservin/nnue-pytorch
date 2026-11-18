"""Gather the existing data-parallel loader's sparse inputs in rank order.

This is an opt-in building block for a feature-transformer-sharded model; it
does not change the ordinary DDP loader or training loop.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist

try:
    from ._cuda_transport import (
        unpack_counted_rows,
        unpack_distributed_rows,
        unpack_native_distributed_rows,
    )
except ImportError:
    unpack_counted_rows = None
    unpack_distributed_rows = None
    unpack_native_distributed_rows = None


@dataclass(frozen=True)
class GlobalFeatureBatch:
    us: torch.Tensor
    them: torch.Tensor
    white_indices: torch.Tensor
    black_indices: torch.Tensor
    layer_stack_indices: torch.Tensor
    # Offsets are in process-group rank order. The local targets remain in the
    # original loader batch, corresponding to [offsets[rank]:offsets[rank+1]].
    offsets: tuple[int, ...]
    rank: int
    # When present, the index tensors contain [start, end) ranges into the
    # flattened payload instead of padded feature IDs.
    packed_payload: torch.Tensor | None = None
    packed_feature_width: int | None = None

    @property
    def local_slice(self) -> slice:
        return slice(self.offsets[self.rank], self.offsets[self.rank + 1])


def _unpack_rows_reference(counts: torch.Tensor, ids: torch.Tensor, width: int) -> torch.Tensor:
    rows = counts.numel()
    result = torch.full((rows, width), -1, dtype=torch.int32, device=counts.device)
    if ids.numel():
        row = torch.repeat_interleave(
            torch.arange(rows, device=counts.device),
            counts.to(torch.int64),
            output_size=ids.numel(),
        )
        starts = counts.to(torch.int64).cumsum(0) - counts
        column = torch.arange(ids.numel(), device=counts.device) - starts[row]
        result[row, column] = ids
    return result


def _unpack_rows(counts: torch.Tensor, ids: torch.Tensor, width: int) -> torch.Tensor:
    if counts.device.type == "cuda" and torch.version.hip is None and unpack_counted_rows:
        return unpack_counted_rows(counts, ids, width)
    return _unpack_rows_reference(counts, ids, width)


def _use_fused_unpack(device):
    return device.type == "cuda" and torch.version.hip is None and unpack_distributed_rows is not None


class PendingNativeFeatures:
    """Own buffers and collective work until an asynchronous gather is consumed."""

    def __init__(self, gathered, dimensions, rank, work, keepalive, ready_event):
        self.gathered = gathered
        self.dimensions = dimensions
        self.rank = rank
        self.work = work
        self.keepalive = keepalive
        self.ready_event = ready_event
        self._result = None

    def finish(self, *, direct_packed_ft=False):
        """Wait once and retain the layout selected by the first call."""
        if self._result is None:
            if self.ready_event is not None:
                torch.cuda.current_stream(self.gathered.device).wait_event(self.ready_event)
            if self.work is not None:
                self.work.wait()
            if self.gathered.is_cuda:
                self.gathered.record_stream(torch.cuda.current_stream(self.gathered.device))
            self._result = _unpack_native_gathered(
                self.gathered, self.dimensions, self.rank,
                direct_packed_ft=direct_packed_ft,
            )
        return self._result


def begin_native_packed_features(batch, group=None):
    """Distribute trusted native offsets; payload contents are not revalidated.

    The metadata and offsets must originate from native packing (or a caller
    that has independently validated the same invariants). All ranks must use
    this format together.
    """
    us, them, native, black, _, _, stacks = batch
    if native.get("format") != "native_offsets_v1" or black is not None:
        raise ValueError("invalid native packed sharded batch")
    packed = native["packed"]
    n, width, nw, nb = (native[k] for k in ("size", "width", "nw", "nb"))
    if (not all(isinstance(x, int) and x >= 0 for x in (n, width, nw, nb))
        or nw > n * width or nb > n * width
        or packed.dtype != torch.int32 or packed.ndim != 1 or not packed.is_contiguous()
        or packed.numel() != 3*n + 2 + nw + nb or us.shape != (n, 1)
        or them.shape != (n, 1) or us.dtype != torch.float32 or them.dtype != torch.float32
        or stacks.shape != (n,) or stacks.dtype != torch.int64
        or any(t.device != packed.device for t in (us, them, stacks))):
        raise ValueError("invalid native packed sharded batch dimensions")
    # The public tuple's stack tensor is authoritative, just as in the dense
    # path; callers may change it after receiving a native packed batch.
    payload = torch.cat((packed[:packed.numel()-n], stacks.to(torch.int32),
                         us.contiguous().view(torch.int32).reshape(-1),
                         them.contiguous().view(torch.int32).reshape(-1)))
    distributed = dist.is_available() and dist.is_initialized()
    world = dist.get_world_size(group) if distributed else 1
    rank = dist.get_rank(group) if distributed else 0
    info = torch.tensor([n, width, nw, nb, 1], dtype=torch.int64, device=packed.device)
    infos = [torch.empty_like(info) for _ in range(world)]
    if distributed:
        dist.all_gather(infos, info, group=group)
    else:
        infos[0] = info
    descriptions = [tuple(map(int, item.tolist())) for item in infos]
    if any(desc[4] != 1 for desc in descriptions):
        raise ValueError("all ranks must use the same feature batch format")
    dimensions = [desc[:4] for desc in descriptions]
    if any(w != width for _, w, _, _ in dimensions):
        raise ValueError("all ranks must use the same maximum feature count")
    length = max(1, *(5*b + 2 + w + k for b, _, w, k in dimensions))
    padded = torch.zeros(length, dtype=torch.int32, device=packed.device)
    padded[:payload.numel()] = payload
    gathered = torch.empty((world, length), dtype=torch.int32, device=packed.device)
    if distributed:
        work = dist.all_gather(list(gathered.unbind(0)), padded, group=group, async_op=True)
    else:
        gathered[0].copy_(padded)
        work = None
    ready_event = None
    if packed.is_cuda:
        ready_event = torch.cuda.Event()
        ready_event.record(torch.cuda.current_stream(packed.device))
    return PendingNativeFeatures(gathered, dimensions, rank, work, (batch, payload, padded, infos), ready_event)


def _unpack_native_gathered(gathered, dimensions, rank, *, direct_packed_ft=False):
    width = dimensions[0][1]
    feature_ranges = False
    if gathered.is_cuda and torch.version.hip is None and unpack_native_distributed_rows is not None:
        feature_ranges = (
            direct_packed_ft and 0 < width and any(b for b, _, _, _ in dimensions)
            and gathered.numel() <= torch.iinfo(torch.int32).max
        )
        result = unpack_native_distributed_rows(
            gathered, dimensions, width, feature_ranges=feature_ranges
        )
    else:
        fields = [[], [], [], [], []]
        for src, (b, _, w, k) in zip(gathered, dimensions):
            wo, bo = src[:b+1], src[b+1:2*(b+1)]
            start = 2 * (b + 1)
            stack_start = start + w + k
            fields[0].append(src[stack_start+b:stack_start+2*b].view(torch.float32).reshape(b, 1))
            fields[1].append(src[stack_start+2*b:stack_start+3*b].view(torch.float32).reshape(b, 1))
            fields[2].append(_unpack_rows_reference(wo[1:] - wo[:-1], src[start:start+w], width))
            fields[3].append(_unpack_rows_reference(bo[1:] - bo[:-1], src[start+w:start+w+k], width))
            fields[4].append(src[stack_start:stack_start+b].to(torch.int64))
        result = tuple(torch.cat(parts) for parts in fields)
    offsets = [0]
    for b, _, _, _ in dimensions:
        offsets.append(offsets[-1] + b)
    return GlobalFeatureBatch(
        *result, tuple(offsets), rank,
        gathered if feature_ranges else None, width if feature_ranges else None,
    )


def _gather_native_packed_features(batch, group=None, *, direct_packed_ft=False):
    return begin_native_packed_features(batch, group).finish(direct_packed_ft=direct_packed_ft)


def gather_packed_features(
    batch: tuple[torch.Tensor, ...], group=None, *, direct_packed_ft=False
) -> GlobalFeatureBatch:
    """Gather one loader batch from each rank, in process-group rank order.

    ``batch`` is the seven-tensor tuple from SparseBatch.get_tensors. Feature
    rows must have the loader's normal trailing -1 padding, and layer-stack
    values must originate from the loader's int32 layer-stack indices. The
    collective uses the input device (CUDA for NCCL, CPU for Gloo). All ranks
    must call this function for each step, even ranks with an empty batch.

    The current implementation packs already-expanded tensors and pads each
    rank's payload to the largest packed payload for all_gather. Variable-size
    packing and size exchange synchronize the CUDA stream; benchmark before
    using this as a performance claim.
    """
    us, them, white, black, _outcome, _score, stacks = batch
    if isinstance(white, dict):
        return _gather_native_packed_features(batch, group, direct_packed_ft=direct_packed_ft)
    n, width = white.shape
    if (
        white.dtype != torch.int32
        or black.dtype != torch.int32
        or black.shape != white.shape
        or us.shape != (n, 1)
        or them.shape != (n, 1)
        or stacks.shape != (n,)
        or us.dtype != torch.float32
        or them.dtype != torch.float32
        or stacks.dtype != torch.int64
    ):
        raise ValueError("batch must have the standard sparse loader tensor layout")
    if any(t.device != white.device for t in (us, them, black, stacks)):
        raise ValueError("feature tensors must be on the same device")

    white_mask, black_mask = white >= 0, black >= 0
    white_ids, black_ids = white[white_mask], black[black_mask]
    white_counts = white_mask.sum(dim=1, dtype=torch.int32)
    black_counts = black_mask.sum(dim=1, dtype=torch.int32)
    payload = torch.cat(
        (
            white_counts,
            black_counts,
            us.contiguous().view(torch.int32).reshape(-1),
            them.contiguous().view(torch.int32).reshape(-1),
            stacks.to(torch.int32),
            white_ids,
            black_ids,
        )
    )
    distributed = dist.is_available() and dist.is_initialized()
    world = dist.get_world_size(group) if distributed else 1
    rank = dist.get_rank(group) if distributed else 0
    info = torch.tensor(
        [n, width, white_ids.numel(), black_ids.numel(), 0],
        dtype=torch.int64,
        device=white.device,
    )
    infos = [torch.empty_like(info) for _ in range(world)]
    if distributed:
        dist.all_gather(infos, info, group=group)
    else:
        infos[0] = info
    descriptions = [tuple(map(int, item.tolist())) for item in infos]
    if any(desc[4] != 0 for desc in descriptions):
        raise ValueError("all ranks must use the same feature batch format")
    dimensions = [desc[:4] for desc in descriptions]
    if any(w != width for _, w, _, _ in dimensions):
        raise ValueError("all ranks must use the same maximum feature count")
    lengths = [5 * b + nw + nb for b, _, nw, nb in dimensions]
    padded_length = max(1, *lengths)
    padded = torch.zeros(padded_length, dtype=torch.int32, device=white.device)
    padded[: payload.numel()] = payload
    fused = _use_fused_unpack(white.device)
    gathered_buffer = torch.empty((world, padded_length), dtype=torch.int32, device=white.device) if fused else None
    gathered = list(gathered_buffer.unbind(0)) if fused else [torch.empty_like(padded) for _ in range(world)]
    if distributed:
        dist.all_gather(gathered, padded, group=group)
    else:
        if fused:
            gathered[0].copy_(padded)
        else:
            gathered[0] = padded

    if fused:
        all_us, all_them, all_white, all_black, all_stacks = unpack_distributed_rows(
            gathered_buffer, dimensions, width
        )
        offsets = [0]
        for b, _, _, _ in dimensions:
            offsets.append(offsets[-1] + b)
        return GlobalFeatureBatch(all_us, all_them, all_white, all_black, all_stacks,
                                  tuple(offsets), rank)

    all_us, all_them, all_white, all_black, all_stacks = [], [], [], [], []
    offsets = [0]
    for packed, (b, _, nw, nb) in zip(gathered, dimensions):
        wc, bc = packed[:b], packed[b : 2 * b]
        all_us.append(packed[2 * b : 3 * b].view(torch.float32).reshape(b, 1))
        all_them.append(packed[3 * b : 4 * b].view(torch.float32).reshape(b, 1))
        all_stacks.append(packed[4 * b : 5 * b].to(torch.int64))
        all_white.append(_unpack_rows(wc, packed[5 * b : 5 * b + nw], width))
        all_black.append(_unpack_rows(bc, packed[5 * b + nw : 5 * b + nw + nb], width))
        offsets.append(offsets[-1] + b)
    return GlobalFeatureBatch(
        torch.cat(all_us),
        torch.cat(all_them),
        torch.cat(all_white),
        torch.cat(all_black),
        torch.cat(all_stacks),
        tuple(offsets),
        rank,
    )

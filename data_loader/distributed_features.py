"""Gather every rank's sparse loader batch, in rank order, for the sharded FT."""

from dataclasses import dataclass

import torch
import torch.distributed as dist

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

    @property
    def local_slice(self) -> slice:
        return slice(self.offsets[self.rank], self.offsets[self.rank + 1])


try:
    from ._cuda_transport import unpack_counted_rows
except ImportError:
    unpack_counted_rows = None


def _unpack_rows(counts: torch.Tensor, ids: torch.Tensor, width: int) -> torch.Tensor:
    """Rows of ``width`` IDs, -1 padded, holding ``counts[i]`` IDs each."""
    if counts.is_cuda and torch.version.hip is None and unpack_counted_rows is not None:
        return unpack_counted_rows(counts, ids, width)
    return _unpack_rows_torch(counts, ids, width)


def _unpack_rows_torch(counts: torch.Tensor, ids: torch.Tensor, width: int) -> torch.Tensor:
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


def gather_packed_features(
    batch: tuple[torch.Tensor, ...], group=None
) -> GlobalFeatureBatch:
    """Gather one loader batch from each rank, in process-group rank order.

    ``batch`` is the seven-tensor tuple from SparseBatch.get_tensors. Feature
    rows must have the loader's normal trailing -1 padding, and layer-stack
    values must originate from the loader's int32 layer-stack indices. The
    collective uses the input device (CUDA for NCCL, CPU for Gloo). All ranks
    must call this function for each step with equal-size batches.

    Each rank packs its active feature IDs and pads the payload to the
    largest rank's size for the all_gather. Exchanging those sizes
    synchronizes the CUDA stream once per call.
    """
    us, them, white, black, _outcome, _score, stacks = batch
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
        [n, width, white_ids.numel(), black_ids.numel()],
        dtype=torch.int64,
        device=white.device,
    )
    infos = [torch.empty_like(info) for _ in range(world)]
    if distributed:
        dist.all_gather(infos, info, group=group)
    else:
        infos[0] = info
    dimensions = [tuple(map(int, item.tolist())) for item in infos]
    if any(b != n or w != width for b, w, _, _ in dimensions):
        raise ValueError("all ranks must use the same batch size and maximum feature count")
    padded_length = max(1, *(5 * b + nw + nb for b, _, nw, nb in dimensions))
    padded = torch.zeros(padded_length, dtype=torch.int32, device=white.device)
    padded[: payload.numel()] = payload
    gathered = torch.empty((world, padded_length), dtype=torch.int32, device=white.device)
    if distributed:
        dist.all_gather(list(gathered.unbind(0)), padded, group=group)
    else:
        gathered[0].copy_(padded)

    offsets = [0]
    for b, _, _, _ in dimensions:
        offsets.append(offsets[-1] + b)
    offsets = tuple(offsets)

    # Unpack every rank's rows with one call per perspective.
    us_parts, them_parts, stack_parts = [], [], []
    white_counts, black_counts, white_ids, black_ids = [], [], [], []
    for packed, (b, _, nw, nb) in zip(gathered, dimensions):
        white_counts.append(packed[:b])
        black_counts.append(packed[b : 2 * b])
        us_parts.append(packed[2 * b : 3 * b].view(torch.float32).reshape(b, 1))
        them_parts.append(packed[3 * b : 4 * b].view(torch.float32).reshape(b, 1))
        stack_parts.append(packed[4 * b : 5 * b].to(torch.int64))
        white_ids.append(packed[5 * b : 5 * b + nw])
        black_ids.append(packed[5 * b + nw : 5 * b + nw + nb])
    return GlobalFeatureBatch(
        torch.cat(us_parts),
        torch.cat(them_parts),
        _unpack_rows(torch.cat(white_counts), torch.cat(white_ids), width),
        _unpack_rows(torch.cat(black_counts), torch.cat(black_ids), width),
        torch.cat(stack_parts),
        offsets,
        rank,
    )

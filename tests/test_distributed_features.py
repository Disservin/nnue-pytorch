"""Exercise packed distribution using the real loader tensor layout."""

import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from data_loader import stream
from data_loader import distributed_features as features
from data_loader.distributed_features import gather_packed_features


def _batch(rank, n, width=7):
    us = (torch.arange(n).reshape(-1, 1) + rank).remainder(2).float()
    white = torch.full((n, width), -1, dtype=torch.int32)
    black = torch.full_like(white, -1)
    for row in range(n):
        wc, bc = (rank + row) % (width + 1), (rank * 2 + row + 1) % (width + 1)
        white[row, :wc] = torch.arange(wc, dtype=torch.int32) + rank * 1000 + row * 10
        black[row, :bc] = torch.arange(bc, dtype=torch.int32) + rank * 2000 + row * 10
    targets = torch.full((n, 1), rank, dtype=torch.float32)
    stacks = torch.arange(n, dtype=torch.int64) + rank * 100
    return us, 1 - us, white, black, targets, targets + 1, stacks


def _check(rank, sizes, device="cpu", width=7):
    batches = [_batch(i, size, width=width) for i, size in enumerate(sizes)]
    batches = [tuple(t.to(device) for t in batch) for batch in batches]
    result = gather_packed_features(batches[rank])
    for value, index in zip(
        (
            result.us,
            result.them,
            result.white_indices,
            result.black_indices,
            result.layer_stack_indices,
        ),
        (0, 1, 2, 3, 6),
    ):
        assert torch.equal(value, torch.cat([batch[index] for batch in batches]))
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    assert result.offsets == tuple(offsets)
    assert result.local_slice == slice(offsets[rank], offsets[rank + 1])


def _worker(rank, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank=rank, world_size=4, timeout=timedelta(seconds=30)
    )
    try:
        for sizes in ([3, 3, 3, 3], [0, 0, 0, 0]):
            _check(rank, sizes)
        # Mismatches are detected by every rank before the payload gather.
        batch = _batch(rank, 2, width=8 if rank == 0 else 7)
        with pytest.raises(ValueError, match="same batch size"):
            gather_packed_features(batch)
        with pytest.raises(ValueError, match="same batch size"):
            gather_packed_features(_batch(rank, 2 + (rank == 1)))
    finally:
        dist.destroy_process_group()


def test_single_process():
    _check(0, [5])
    _check(0, [0])


def test_native_loader_batch():
    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "8/8/8/4k3/8/3K4/8/8 b - - 0 1",
    ]
    ptr = stream.get_sparse_batch_from_fens(
        "Full_Threats+PP_3Wide+K32H3", fens, [0, 10], [1, 0], [0, 1]
    )
    try:
        batch = ptr.contents.get_tensors("cpu")
        result = gather_packed_features(batch)
        for actual, index in zip(
            (
                result.us,
                result.them,
                result.white_indices,
                result.black_indices,
                result.layer_stack_indices,
            ),
            (0, 1, 2, 3, 6),
        ):
            assert torch.equal(actual, batch[index])
    finally:
        stream.destroy_sparse_batch(ptr)


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Gloo is unavailable",
)
def test_four_rank_distribution():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=4, join=True)


def _nccl_worker(rank, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=4, timeout=timedelta(seconds=60))
    try:
        for sizes in ([3, 3, 3, 3], [0, 0, 0, 0]):
            _check(rank, sizes, device=torch.device("cuda", rank))
        _check(rank, [2, 2, 2, 2], device=torch.device("cuda", rank), width=0)
        batches = []
        for i, n in enumerate([2, 2, 2, 2]):
            batch = list(_batch(i, n))
            batch[0] = torch.full((n, 1), 0.125 + i * 0.25)
            batch[1] = torch.full((n, 1), -3.25 - i)
            batch[6] = torch.arange(n, dtype=torch.int64) - i - 10
            batches.append(tuple(t.cuda(rank) for t in batch))
        result = gather_packed_features(batches[rank])
        for actual, index in ((result.us, 0), (result.them, 1),
                              (result.white_indices, 2), (result.black_indices, 3),
                              (result.layer_stack_indices, 6)):
            assert torch.equal(actual, torch.cat([b[index] for b in batches]))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_nccl_available() or torch.cuda.device_count() < 4,
    reason="Four CUDA GPUs and NCCL required",
)
def test_four_rank_cuda_distribution():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_nccl_worker, args=(port,), nprocs=4, join=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_single_process_cuda():
    _check(0, [5], device="cuda")
    _check(0, [0], device="cuda")
    _check(0, [3], device="cuda", width=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None
    or features.unpack_counted_rows is None,
    reason="NVIDIA CUDA and Triton required",
)
@pytest.mark.parametrize("counts", [[], [0, 0], [0, 3, 7, 1, 0], [7, 7, 7]])
def test_cuda_unpack_rows_matches_torch(counts):
    sizes = torch.tensor(counts, dtype=torch.int32, device="cuda")
    ids = torch.arange(sum(counts), dtype=torch.int32, device="cuda") + 11
    torch.testing.assert_close(
        features._unpack_rows(sizes, ids, 7),
        features._unpack_rows_torch(sizes, ids, 7),
        rtol=0, atol=0,
    )

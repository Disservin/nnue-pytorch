"""Exercise packed distribution using the real loader tensor layout."""

import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from data_loader import stream
from data_loader.distributed_features import (
    _unpack_rows,
    _unpack_rows_reference,
    gather_packed_features,
    unpack_counted_rows,
    unpack_distributed_rows,
)


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
        for sizes in ([3, 3, 3, 3], [1, 0, 4, 2], [0, 0, 0, 0]):
            _check(rank, sizes)
        # Width mismatch is detected by every rank before the payload gather.
        batch = _batch(rank, 2, width=8 if rank == 0 else 7)
        with pytest.raises(ValueError, match="same maximum"):
            gather_packed_features(batch)
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
        for sizes in ([3, 3, 3, 3], [1, 0, 4, 2], [0, 0, 0, 0]):
            _check(rank, sizes, device=torch.device("cuda", rank))
        _check(rank, [3, 0, 1, 2], device=torch.device("cuda", rank), width=0)
        batches = []
        for i, n in enumerate([2, 0, 3, 1]):
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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None or unpack_counted_rows is None,
    reason="NVIDIA CUDA and Triton are required",
)
@pytest.mark.parametrize("counts", [[], [0, 0], [0, 3, 7, 1, 0], [7, 7, 7]])
def test_cuda_unpack_rows_matches_reference_on_nondefault_stream(counts):
    width = 7
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        sizes = torch.tensor(counts, dtype=torch.int32, device="cuda")
        ids = torch.arange(sum(counts), dtype=torch.int32, device="cuda") + 11
        expected = _unpack_rows_reference(sizes, ids, width)
        actual = _unpack_rows(sizes, ids, width)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    stream.synchronize()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or torch.version.hip is not None or unpack_counted_rows is None,
    reason="Two NVIDIA GPUs and Triton are required",
)
def test_cuda_unpack_noncurrent_device_and_zero_width():
    torch.cuda.set_device(0)
    sizes = torch.tensor([0, 2, 0], dtype=torch.int32, device="cuda:1")
    ids = torch.tensor([3, 7], dtype=torch.int32, device="cuda:1")
    torch.testing.assert_close(_unpack_rows(sizes, ids, 3),
                               _unpack_rows_reference(sizes, ids, 3), rtol=0, atol=0)
    empty = torch.zeros(3, dtype=torch.int32, device="cuda:1")
    assert _unpack_rows(empty, ids[:0], 0).shape == (3, 0)
    assert torch.cuda.current_device() == 0


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or torch.version.hip is not None or unpack_distributed_rows is None,
    reason="Two NVIDIA GPUs and Triton are required",
)
@pytest.mark.parametrize("width", [0, 7])
@pytest.mark.parametrize("streamed", [False, True])
def test_fused_unpack_empty_rows_noncurrent_device(width, streamed):
    torch.cuda.set_device(0)
    batches = [_batch(i, n, width=width) for i, n in enumerate([3, 0, 2])]
    packed_rows, dims = [], []
    for batch in batches:
        u, v, white, black, _, _, stacks = batch
        white.fill_(-1)
        black.fill_(-1)
        n = len(u)
        dims.append((n, width, 0, 0))
        packed_rows.append(torch.cat((torch.zeros(2 * n, dtype=torch.int32),
                                      u.flatten().view(torch.int32),
                                      v.flatten().view(torch.int32), stacks.int())))
    stride = max(1, *(len(p) for p in packed_rows))
    payload = torch.zeros((len(batches), stride), dtype=torch.int32)
    for rank, row in enumerate(packed_rows):
        payload[rank, :len(row)] = row
    payload = payload.to("cuda:1")
    stream = torch.cuda.Stream(device=1) if streamed else torch.cuda.current_stream(1)
    stream.wait_stream(torch.cuda.current_stream(1))
    with torch.cuda.stream(stream):
        output = unpack_distributed_rows(payload, dims, width)
        for actual, index in zip(output, (0, 1, 2, 3, 6)):
            expected = torch.cat([b[index] for b in batches]).to("cuda:1")
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    stream.synchronize()
    assert torch.cuda.current_device() == 0


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None
                    or unpack_distributed_rows is None, reason="NVIDIA CUDA and Triton required")
def test_variable_payload_lengths_do_not_recompile_decoder():
    from data_loader._cuda_transport import _unpack_distributed_kernel
    if not hasattr(_unpack_distributed_kernel, "device_caches"):
        pytest.skip("Triton does not expose a per-device kernel cache")
    torch.cuda.set_device(0)
    batch = tuple(t.cuda() for t in _batch(0, 3))
    u, v, white, black, _, _, stacks = batch
    wm, bm = white >= 0, black >= 0
    wi, bi = white[wm], black[bm]
    dimensions = [(3, 7, wi.numel(), bi.numel())]
    dense_payload = torch.cat((wm.sum(1, dtype=torch.int32), bm.sum(1, dtype=torch.int32),
                               u.flatten().view(torch.int32), v.flatten().view(torch.int32),
                               stacks.int(), wi, bi))
    for kernel, fn, source in (
        (_unpack_distributed_kernel, unpack_distributed_rows, dense_payload),
    ):
        expected_cache_size = None
        for extra in (0, 1, 17, 257):
            payload = torch.zeros((1, source.numel() + extra), dtype=torch.int32, device="cuda")
            payload[0, :source.numel()] = source
            result = fn(payload, dimensions, 7)
            torch.testing.assert_close(result[2], white, rtol=0, atol=0)
            torch.testing.assert_close(result[3], black, rtol=0, atol=0)
            cache_size = len(kernel.device_caches[torch.cuda.current_device()][0])
            if expected_cache_size is None:
                expected_cache_size = cache_size
            else:
                assert cache_size == expected_cache_size
        for n in (2, 4):
            changed = tuple(t.cuda() for t in _batch(0, n))
            cu, cv, cw, cb, _, _, cs = changed
            cwm, cbm = cw >= 0, cb >= 0
            cwi, cbi = cw[cwm], cb[cbm]
            new_dims = [(n, 7, cwi.numel(), cbi.numel())]
            payload = torch.cat((cwm.sum(1, dtype=torch.int32), cbm.sum(1, dtype=torch.int32),
                                 cu.flatten().view(torch.int32), cv.flatten().view(torch.int32),
                                 cs.int(), cwi, cbi))
            result = fn(payload.unsqueeze(0), new_dims, 7)
            torch.testing.assert_close(result[2], cw, rtol=0, atol=0)
            torch.testing.assert_close(result[3], cb, rtol=0, atol=0)
            assert len(kernel.device_caches[torch.cuda.current_device()][0]) == expected_cache_size

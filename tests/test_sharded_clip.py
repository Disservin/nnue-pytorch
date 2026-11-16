"""Check batched sharded clipping against the original scalar implementation."""
import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from model.sharded_backward import clip_sharded_grad_norm_


@torch.no_grad()
def _scalar_clip(sharded, replicated, max_norm):
    params = sharded + replicated
    total = torch.zeros((), dtype=torch.float64, device=params[0].device)
    for p in sharded:
        if p.grad is not None:
            total += torch.linalg.vector_norm(p.grad).to(torch.float64).square()
    dist.all_reduce(total)
    for p in replicated:
        if p.grad is not None:
            total += torch.linalg.vector_norm(p.grad).to(torch.float64).square()
    norm = total.sqrt()
    scale = (max_norm / (norm + 1e-6)).clamp(max=1.0)
    for p in params:
        if p.grad is not None:
            p.grad.mul_(scale.to(dtype=p.grad.dtype))
    return norm


def _worker(rank, port, backend):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend, rank=rank, world_size=4, timeout=timedelta(seconds=120))
    try:
        for max_norm, zero, absent in ((0.2, False, False), (1e8, False, False),
                                       (0.2, True, False), (0.2, False, True)):
            torch.manual_seed(42 + rank)
            shards = [torch.nn.Parameter(torch.empty(23, 17, device=device)),
                      torch.nn.Parameter(torch.empty(13, device=device, dtype=torch.float64))]
            replicas = [torch.nn.Parameter(torch.empty(7, 11, device=device)),
                        torch.nn.Parameter(torch.empty(3, device=device))]
            for p in shards:
                p.grad = torch.randn_like(p) if not zero else torch.zeros_like(p)
            torch.manual_seed(88)
            replicas[0].grad = (torch.randn_like(replicas[0]) if not zero
                                else torch.zeros_like(replicas[0]))
            if absent:
                # A rank with no gradients must still enter the norm all-reduce.
                replicas[0].grad = None
                if rank == 0:
                    for p in shards:
                        p.grad = None
            # Exercise a parameter whose gradient is absent on every rank.
            copies = [torch.nn.Parameter(p.detach().clone()) for p in shards + replicas]
            for p, q in zip(shards + replicas, copies):
                q.grad = None if p.grad is None else p.grad.clone()
            expected = _scalar_clip(shards, replicas, max_norm)
            actual = clip_sharded_grad_norm_(iter(copies[:2]), iter(copies[2:]), max_norm)
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
            for p, q in zip(shards + replicas, copies):
                if p.grad is None:
                    assert q.grad is None
                else:
                    torch.testing.assert_close(q.grad, p.grad, rtol=1e-6, atol=1e-7)
        # All ranks with all gradients absent still participate and return zero.
        p = torch.nn.Parameter(torch.empty(2, device=device))
        assert clip_sharded_grad_norm_([p], [], 1.0).item() == 0.0
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("backend", ["gloo", "nccl"])
def test_foreach_clipping_matches_reference(backend):
    if backend == "gloo" and not dist.is_gloo_available():
        pytest.skip("Gloo is unavailable")
    if backend == "nccl" and (not dist.is_nccl_available() or torch.cuda.device_count() < 4):
        pytest.skip("requires four CUDA devices")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, backend), nprocs=4, join=True)


def test_clip_validation_and_no_gradients():
    p = torch.nn.Parameter(torch.ones(3))
    assert clip_sharded_grad_norm_([p], [], 0).item() == 0
    with pytest.raises(ValueError, match="at least one"):
        clip_sharded_grad_norm_([], [], 1)
    with pytest.raises(ValueError, match="duplicates"):
        clip_sharded_grad_norm_([p], [p], 1)
    with pytest.raises(ValueError, match="nonnegative"):
        clip_sharded_grad_norm_([p], [], -1)


def test_analytical_norm_noncontiguous_and_empty():
    shard = torch.nn.Parameter(torch.zeros(2, 3))
    shard.grad = torch.tensor([[3., 0.], [4., 0.], [0., 0.]]).t()
    assert not shard.grad.is_contiguous()
    empty = torch.nn.Parameter(torch.empty(0))
    empty.grad = torch.empty(0)
    replica = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    replica.grad = torch.tensor([12.], dtype=torch.float64)
    norm = clip_sharded_grad_norm_([shard, empty], [replica], 5.)
    torch.testing.assert_close(norm, torch.tensor(13., dtype=torch.float64))
    torch.testing.assert_close(replica.grad, torch.tensor([12 * 5 / (13 + 1e-6)], dtype=torch.float64))
    assert empty.grad.numel() == 0
    clip_sharded_grad_norm_([shard, empty], [replica], 0.)
    assert not torch.count_nonzero(shard.grad)
    assert not torch.count_nonzero(replica.grad)

"""Check the experimental sharded clip against the existing implementation."""
import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from model.sharded_backward import clip_sharded_grad_norm_
from scripts.experimental_foreach_clip import foreach_sharded_clip


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
        for max_norm, zero in ((0.2, False), (1e8, False), (0.2, True)):
            torch.manual_seed(42 + rank)
            shards = [torch.nn.Parameter(torch.empty(23, 17, device=device)),
                      torch.nn.Parameter(torch.empty(13, device=device))]
            replicas = [torch.nn.Parameter(torch.empty(7, 11, device=device)),
                        torch.nn.Parameter(torch.empty(3, device=device))]
            for p in shards:
                p.grad = torch.randn_like(p) if not zero else torch.zeros_like(p)
            torch.manual_seed(88)
            replicas[0].grad = (torch.randn_like(replicas[0]) if not zero
                                else torch.zeros_like(replicas[0]))
            # Exercise a parameter whose gradient is absent on every rank.
            copies = [torch.nn.Parameter(p.detach().clone()) for p in shards + replicas]
            for p, q in zip(shards + replicas, copies):
                q.grad = None if p.grad is None else p.grad.clone()
            expected = clip_sharded_grad_norm_(shards, replicas, max_norm)
            actual = foreach_sharded_clip(copies[:2], copies[2:], max_norm)
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
            for p, q in zip(shards + replicas, copies):
                if p.grad is None:
                    assert q.grad is None
                else:
                    torch.testing.assert_close(q.grad, p.grad, rtol=1e-6, atol=1e-7)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("backend", ["gloo", "nccl"])
def test_foreach_clipping_matches_reference(backend):
    if backend == "nccl" and torch.cuda.device_count() < 4:
        pytest.skip("requires four CUDA devices")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, backend), nprocs=4, join=True)

"""The diagnostic FT ablation must preserve other gradient reductions."""
import copy
import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from scripts.ddp_comm_diagnostics import make_diagnostic_hook


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(2, 2, bias=False)
        self.tail = nn.Linear(2, 1, bias=False)

    def forward(self, x):
        return self.tail(self.input(x))


def _worker(rank, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        for mode in ("hook", "skip-ft", "skip-all"):
            torch.manual_seed(17)
            model = _Model()
            reference = copy.deepcopy(model)
            ddp = DistributedDataParallel(model, static_graph=True, gradient_as_bucket_view=True)
            info = {}
            ddp.register_comm_hook(dist.group.WORLD, make_diagnostic_hook(model, mode, info))
            x = torch.tensor([[float(rank + 1), float(rank + 2)]])
            ddp(x).sum().backward()
            reference(x).sum().backward()
            local_ft = reference.input.weight.grad.clone()
            local_tail = reference.tail.weight.grad.clone()
            expected_ft = local_ft.clone()
            expected_tail = local_tail.clone()
            dist.all_reduce(expected_ft)
            dist.all_reduce(expected_tail)
            expected_ft /= 2
            expected_tail /= 2
            torch.testing.assert_close(model.input.weight.grad,
                                       expected_ft if mode == "hook" else local_ft)
            torch.testing.assert_close(model.tail.weight.grad,
                                       local_tail if mode == "skip-all" else expected_tail)
            assert sum(v["ft_bytes"] for v in info.values()) == model.input.weight.numel() * 4
            assert sum(v["other_bytes"] for v in info.values()) == model.tail.weight.numel() * 4
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo required")
def test_ddp_diagnostic_gradient_semantics():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=2, join=True)

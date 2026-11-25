"""Two-GPU production-shape smoke test of the real NNUE feature components."""

import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from model.config import LossParams, ModelConfig, NNUEConfig
from model.nnue import NNUE
from model.optimizers import OptimizerConfig
from trainer.sharded_adamw import ShardedAdamWStep


def _batch(rank, n):
    white = torch.full((n, 260), -1, dtype=torch.int32)
    black = torch.full_like(white, -1)
    for row in range(n):
        white[row, :3] = torch.tensor([5 + row, 59808 + 10 + row, 64368 + 123 + row])
        black[row, :3] = torch.tensor([25 + row, 59808 + 20 + row, 64368 + 456 + row])
    us = ((torch.arange(n) + rank) % 2).float().reshape(-1, 1)
    outcome = (torch.arange(n) % 2).float().reshape(-1, 1)
    score = (torch.arange(n).float().reshape(-1, 1) * 300) - 700 + rank * 200
    stacks = (torch.arange(n, dtype=torch.int64) * 31 + rank * 47) % 192
    return us, 1 - us, white, black, outcome, score, stacks


def _worker(rank, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl", rank=rank, world_size=2, timeout=timedelta(seconds=180)
    )
    try:
        torch.set_num_threads(4)
        config = NNUEConfig(
            model_config=ModelConfig(L1=1024, L2=32, L3=32),
            optimizer_config=OptimizerConfig(optimizer_name="adamw", adamw_fused=False),
            loss_params=LossParams(w1=5),
        )
        torch.manual_seed(99)
        full = NNUE(config, max_epoch=2, num_batches_per_epoch=2)
        sharded = NNUE(config, max_epoch=2, num_batches_per_epoch=2, sharding=(rank, 2))
        sharded.model.input.load_unsharded(full.model.input)
        sharded.model.layer_stacks.l1.load_unsharded(full.model.layer_stacks.l1)
        sharded.model.layer_stacks.l2.load_state_dict(
            full.model.layer_stacks.l2.state_dict()
        )
        sharded.model.layer_stacks.output.load_state_dict(
            full.model.layer_stacks.output.state_dict()
        )
        full.to(device)
        sharded.to(device)
        reference_optimizer = full.configure_optimizers()[0][0]
        runner = ShardedAdamWStep(sharded, gradient_clip_val=0.2)
        parts = [_batch(0, 3), _batch(1, 3)]
        global_batch = tuple(
            torch.cat([p[i] for p in parts]).to(device) for i in range(7)
        )
        local_batch = tuple(t.to(device) for t in parts[rank])
        reference_optimizer.zero_grad(set_to_none=True)
        loss = full.compute_loss(global_batch, 0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(full.model.parameters(), 0.2)
        reference_optimizer.step()
        got = runner.step(local_batch)
        torch.testing.assert_close(got, loss.detach(), atol=2e-5, rtol=1e-4)
        expected = dict(full.model.named_parameters())
        for name, param in sharded.model.named_parameters():
            reference = expected[name]
            if name.startswith("input.") or (
                name.startswith("layer_stacks.l1.") and param.ndim == 2
            ):
                reference = sharded.model.input.layout.select_columns(reference)
            torch.testing.assert_close(param, reference, atol=2e-5, rtol=1e-4)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available()
    or not dist.is_nccl_available()
    or torch.cuda.device_count() < 2,
    reason="Two CUDA GPUs and NCCL required",
)
def test_two_gpu_real_network():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=2, join=True)

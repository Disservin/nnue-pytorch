"""Two-GPU NCCL smoke test for the actual sharded AdamW step."""

import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_sharded_adamw import _batch, _config
from test_sharded_ft import _Feature

import model.model as model_module
import model.sharded_model as sharded_model_module
from model.nnue import NNUE
from trainer.sharded_adamw import ShardedAdamWStep


def _worker(rank, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl", rank=rank, world_size=2, timeout=timedelta(seconds=120)
    )
    try:
        model_module.get_feature_cls = lambda _: [_Feature]
        sharded_model_module.get_feature_cls = lambda _: [_Feature]
        config = _config()
        config.model_config.L1 = 64
        torch.manual_seed(42)
        full = NNUE(config, max_epoch=3, num_batches_per_epoch=5)
        with torch.no_grad():
            full.model.input.bias.uniform_(0.25, 0.4)
            for name, param in full.model.layer_stacks.named_parameters():
                if "factorized" in name or "group_factor" in name:
                    param.uniform_(-0.05, 0.05)
        sharded = NNUE(config, max_epoch=3, num_batches_per_epoch=5, sharding=(rank, 2))
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
        for step, sizes in enumerate(([2, 3], [0, 5])):
            parts = [_batch(i, n, step) for i, n in enumerate(sizes)]
            global_batch = tuple(
                torch.cat([p[i] for p in parts]).to(device) for i in range(7)
            )
            local_batch = tuple(t.to(device) for t in parts[rank])
            reference_optimizer.zero_grad(set_to_none=True)
            loss = full.compute_loss(global_batch, step)
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
def test_two_gpu_adamw():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=2, join=True)

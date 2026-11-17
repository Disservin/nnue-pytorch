"""Four-rank QAT/AdamW smoke test for the shared-L1-factor ablation."""
import copy
import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import model.model as model_module
import model.sharded_model as sharded_model_module
from model.nnue import NNUE
from scripts.experimental_no_l1_factor import apply_no_shared_l1_factor
from test_sharded_adamw import _batch, _config
from test_sharded_ft import _Feature
from trainer.sharded_adamw import ShardedAdamWStep


def _worker(rank, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", rank=rank, world_size=4, timeout=timedelta(seconds=180))
    try:
        model_module.get_feature_cls = lambda _: [_Feature]
        sharded_model_module.get_feature_cls = lambda _: [_Feature]
        config = _config()
        torch.manual_seed(125 + rank)
        reference = NNUE(config, max_epoch=3, num_batches_per_epoch=5, sharding=(rank, 4)).cuda(rank)
        baseline = ShardedAdamWStep(reference, gradient_clip_val=None)
        initial = copy.deepcopy(reference.model.state_dict())
        batch = tuple(t.cuda(rank) for t in _batch(rank, [2, 0, 3, 1][rank], 0))
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(rank)
        expected_loss = baseline.step(batch)
        expected = {name: (p.detach().clone(), None if p.grad is None else p.grad.clone(),
                           copy.deepcopy(baseline.optimizer.state.get(p, {})))
                    for name, p in reference.model.named_parameters()}

        candidate = NNUE(copy.deepcopy(config), max_epoch=3, num_batches_per_epoch=5,
                         sharding=(rank, 4)).cuda(rank)
        candidate.model.load_state_dict(initial)
        apply_no_shared_l1_factor(candidate)
        runner = ShardedAdamWStep(candidate, gradient_clip_val=None)
        runner.sharded = [p for p in runner.sharded if p.requires_grad]
        runner.replicated = [p for p in runner.replicated if p.requires_grad]
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, rank)
        actual_loss = runner.step(batch)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=2e-5, atol=2e-6)
        optimized = {id(p) for group in runner.optimizer.param_groups for p in group["params"]}
        for name, p in candidate.model.named_parameters():
            if name.startswith("layer_stacks.l1.factorized_linear."):
                assert not p.requires_grad and p.grad is None and id(p) not in optimized
                continue
            weight, grad, state = expected[name]
            if grad is None:
                assert p.grad is None
            else:
                torch.testing.assert_close(p.grad, grad, rtol=3e-4, atol=5e-6)
            torch.testing.assert_close(p, weight, rtol=3e-4, atol=5e-6)
            for key, value in state.items():
                actual = runner.optimizer.state[p][key]
                if torch.is_tensor(value):
                    torch.testing.assert_close(actual, value, rtol=3e-4, atol=5e-6)
                else:
                    assert actual == value
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_nccl_available() or torch.cuda.device_count() < 4,
                    reason="Four CUDA GPUs and NCCL required")
def test_no_shared_l1_factor_four_rank_step():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=4, join=True)

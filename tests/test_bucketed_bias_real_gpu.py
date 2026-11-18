"""Four-rank parity on the real remapped model and saved global batch."""
import copy
import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from model.modules.bucketed_bias import _eligible
from model.nnue import NNUE
from scripts.benchmark_sharded_adamw import pack_saved_batch
from scripts.nettest_benchmark_settings import make_nettest_settings
from trainer.sharded_adamw import ShardedAdamWStep


def _worker(rank, port, remap_path, batch_dir):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", rank=rank, world_size=4, timeout=timedelta(seconds=240))
    try:
        with open(remap_path, encoding="utf-8") as f:
            config, _, _ = make_nettest_settings(f.read().strip())
        torch.manual_seed(41 + rank)
        baseline = NNUE(config, max_epoch=560, num_batches_per_epoch=8192, sharding=(rank, 4)).cuda(rank)
        reference = ShardedAdamWStep(baseline, bucketed_bias=False)
        candidate = NNUE(copy.deepcopy(config), max_epoch=560, num_batches_per_epoch=8192,
                         sharding=(rank, 4)).cuda(rank)
        optimized = ShardedAdamWStep(candidate, bucketed_bias=True)
        cfg = config.loss_params.lambda_config
        reference.global_step = int(cfg.lambda_schedule_steps * cfg.lambda_cycle_warmup_pct)
        optimized.load_state_dict(reference.state_dict())
        filename = os.path.join(batch_dir, f"batch-0-{rank}-131072.pt")
        batch = pack_saved_batch(torch.load(filename, map_location=torch.device("cuda", rank), weights_only=True))
        l1 = candidate.model.layer_stacks.l1
        assert l1.count == 92
        probe = torch.empty((batch[-1].numel(), l1.out_features), device=rank)
        assert _eligible(probe, l1.linear.bias, batch[-1], l1.count, l1.out_features)
        del probe
        for step in range(2):
            torch.manual_seed(400 + step + rank)
            expected_loss = reference.step(batch)
            torch.manual_seed(400 + step + rank)
            actual_loss = optimized.step(batch)
            torch.testing.assert_close(actual_loss, expected_loss, rtol=3e-4, atol=2e-6)
            jitter = candidate.lambda_scheduler.jitter_buffer
            by_rank = torch.empty(4, device=rank)
            dist.all_gather_into_tensor(by_rank, jitter)
            torch.testing.assert_close(by_rank, baseline.lambda_scheduler.jitter_buffer.expand(4),
                                       atol=0, rtol=0)
            for (name, p), (candidate_name, q) in zip(baseline.model.named_parameters(),
                                                       candidate.model.named_parameters()):
                assert name == candidate_name
                if p.grad is None:
                    assert q.grad is None
                else:
                    torch.testing.assert_close(q.grad, p.grad, rtol=3e-4, atol=5e-6,
                                               msg=lambda msg: f"{name} grad: {msg}")
                torch.testing.assert_close(q, p, rtol=3e-4, atol=5e-6,
                                           msg=lambda msg: f"{name} parameter: {msg}")
                for key, value in reference.optimizer.state.get(p, {}).items():
                    actual = optimized.optimizer.state[q][key]
                    if torch.is_tensor(value):
                        torch.testing.assert_close(actual, value, rtol=3e-4, atol=5e-6)
                    else:
                        assert actual == value
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_nccl_available() or torch.cuda.device_count() < 4
                    or not os.environ.get("NNUE_TEST_REAL_REMAP_CACHE")
                    or not os.environ.get("NNUE_TEST_REAL_BATCH_CACHE"),
                    reason="four GPUs and real remap/batch caches required")
def test_bucketed_bias_real_model_four_rank():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, os.environ["NNUE_TEST_REAL_REMAP_CACHE"],
                            os.environ["NNUE_TEST_REAL_BATCH_CACHE"]), nprocs=4, join=True)

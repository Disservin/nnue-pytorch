"""Check that one-batch-ahead packed distribution preserves AdamW updates."""

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
from data_loader.distributed_features import begin_native_packed_features
from model.nnue import NNUE
from trainer.sharded_adamw import ShardedAdamWStep
from test_distributed_features import _pack_dense_batch
from test_sharded_adamw import _batch, _config
from test_sharded_ft import _Feature


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
        model_module.get_feature_cls = lambda _: [_Feature]
        sharded_model_module.get_feature_cls = lambda _: [_Feature]
        torch.manual_seed(42 + rank)
        nnue = NNUE(_config(), max_epoch=3, num_batches_per_epoch=5, sharding=(rank, 4)).to(device)
        runner = ShardedAdamWStep(nnue, gradient_clip_val=0.2, compile_model=False)
        if os.environ.get("NNUE_TEST_FAST_L1_COLLECTIVE") == "1":
            import model.modules.sharded_l1 as sharded_l1
            from scripts.reference_l1_collective import ReferenceReduceScatter
            original_collective = sharded_l1._SumReduceScatter
            sharded_l1._SumReduceScatter = ReferenceReduceScatter
        initial = copy.deepcopy(runner.state_dict())
        batches = [tuple(t.to(device) for t in _batch(rank, sizes[rank], i))
                   for i, sizes in enumerate(([2, 0, 3, 1], [1, 2, 0, 4],
                                              [0, 1, 4, 1], [1, 1, 1, 1]))]
        expected_losses = [runner.step(batch).clone() for batch in batches]
        expected = copy.deepcopy(runner.state_dict())
        if os.environ.get("NNUE_TEST_COMPILED_DEFAULT") == "1":
            nnue = NNUE(_config(), max_epoch=3, num_batches_per_epoch=5,
                        sharding=(rank, 4)).to(device)
            runner = ShardedAdamWStep(nnue, gradient_clip_val=0.2)
            assert runner.compile_model == (device.type == "cuda")
        runner.load_state_dict(initial)
        if os.environ.get("NNUE_TEST_COMPILE_L1_MERGE") == "1":
            l1 = nnue.model.layer_stacks.l1
            l1._compile_l1_merge = True
        if os.environ.get("NNUE_TEST_FAST_L1_COLLECTIVE") == "1":
            sharded_l1._SumReduceScatter = original_collective
        if os.environ.get("NNUE_ASYNC_TEST_COMPILE") == "1":
            nnue.model.layer_stacks.forward_from_l1 = torch.compile(
                nnue.model.layer_stacks.forward_from_l1
            )
            nnue.enable_compiled_loss()
        packed = [_pack_dense_batch(batch) for batch in batches]
        stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        if stream is not None:
            stream.wait_stream(torch.cuda.current_stream(device))

        def begin(index):
            if stream is None:
                return begin_native_packed_features(packed[index])
            with torch.cuda.stream(stream):
                return begin_native_packed_features(packed[index])

        pending = begin(0)
        for i, batch in enumerate(packed):
            features = pending.finish()
            future = []
            def prepare_next():
                if i + 1 < len(packed):
                    future.append(begin(i + 1))
            actual = runner.step(batch, global_batch=features, after_forward=prepare_next)
            torch.testing.assert_close(actual, expected_losses[i], rtol=1e-5, atol=2e-6)
            pending = future[0] if future else None
        actual = runner.state_dict()
        for name, value in expected["nnue"].items():
            if torch.is_tensor(value):
                torch.testing.assert_close(actual["nnue"][name], value, rtol=3e-5, atol=4e-6)
        expected_optimizer = expected["nnue"]["optimizer_state_dict"]["state"]
        actual_optimizer = actual["nnue"]["optimizer_state_dict"]["state"]
        for parameter_id, states in expected_optimizer.items():
            for key, value in states.items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(actual_optimizer[parameter_id][key], value,
                                               rtol=3e-5, atol=4e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("backend", ["gloo", "nccl"])
def test_four_rank_async_sharded_step(backend):
    if backend == "gloo" and not dist.is_gloo_available():
        pytest.skip("Gloo is unavailable")
    if backend == "nccl" and (not dist.is_nccl_available() or torch.cuda.device_count() < 4):
        pytest.skip("Four CUDA GPUs and NCCL are required")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, backend), nprocs=4, join=True)

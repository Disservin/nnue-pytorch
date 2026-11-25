"""ShardedTrainer steps, checkpoints and resumes; train.py picks the mode."""

import os
import socket
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_sharded_adamw import _batch, _config
from test_sharded_ft import _Feature

import model.model as model_module
import model.sharded_model as sharded_model_module
import train
from model.model import NNUEModel
from model.nnue import NNUE
from trainer.sharded_adamw import ShardedAdamWStep
from trainer.sharded_trainer import ShardedTrainer, is_sharded_checkpoint


def _trainer(rank, root):
    torch.manual_seed(7)
    nnue = NNUE(_config(), max_epoch=2, num_batches_per_epoch=4, sharding=(rank, 2))
    return ShardedTrainer(
        step_runner=ShardedAdamWStep(nnue, gradient_clip_val=2.0),
        model=nnue,
        max_epochs=2,
        check_val_every_n_epoch=1,
        gradient_clip_val=2.0,
        log_every_n_steps=1,
        default_root_dir=root,
        callbacks=[],
        logger=None,
        device="cpu",
        rank=rank,
        world_size=2,
        local_rank=rank,
    )


def _worker(rank, port, root):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        model_module.get_feature_cls = lambda _: [_Feature]
        sharded_model_module.get_feature_cls = lambda _: [_Feature]
        batches = [_batch(rank, 3, step) for step in range(3)]

        trainer = _trainer(rank, root)
        assert trainer._ddp_model is None
        for batch in batches[:2]:
            trainer._optimization_step(batch)
        assert trainer.global_step == 2
        path = os.path.join(root, "last.ckpt")
        trainer.save_checkpoint(path)
        dist.barrier()
        expected = trainer._optimization_step(batches[2])["loss"]

        assert os.path.exists(f"{path}.rank{rank}")
        assert is_sharded_checkpoint(path)
        if rank == 0:
            # The merged checkpoint loads into a plain full model.
            state = torch.load(path, weights_only=False)["state_dict"]
            full = NNUEModel("Test", _config().model_config)
            full.load_state_dict({
                k.removeprefix("model."): v for k, v in state.items()
                if k.startswith("model.")
            })

        resumed = _trainer(rank, root)
        resumed.load_checkpoint(path)
        assert resumed.global_step == 2
        got = resumed._optimization_step(batches[2])["loss"]
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

        resumed.model.eval()
        val = resumed._validation_step(batches[0])["val_loss"]
        assert torch.isfinite(val)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="gloo required"
)
def test_two_rank_checkpoint_resume(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, str(tmp_path)), nprocs=2, join=True)


def _args(**overrides):
    config = _config()
    values = dict(
        multi_gpu="auto", nnue_config=config, swa_start_epoch=-1, max_epochs=10,
        resume_from_checkpoint=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_auto_shards_supported_multi_gpu_runs():
    assert train._use_sharding(_args(), "cuda", 2)
    assert not train._use_sharding(_args(), "cuda", 1)
    assert not train._use_sharding(_args(multi_gpu="ddp"), "cuda", 2)


@pytest.mark.parametrize(
    ("change", "accelerator", "world_size", "reason"),
    [
        ({}, "cpu", 2, "needs CUDA"),
        ({"swa_start_epoch": 3}, "cuda", 2, "SWA"),
        ({}, "cuda", 3, "not divisible"),
    ],
)
def test_auto_falls_back_to_ddp(change, accelerator, world_size, reason, capsys):
    assert not train._use_sharding(_args(**change), accelerator, world_size)
    assert reason in capsys.readouterr().out
    with pytest.raises(ValueError, match=reason):
        train._use_sharding(_args(multi_gpu="sharded", **change), accelerator, world_size)


def test_auto_falls_back_for_other_optimizers():
    args = _args()
    args.nnue_config.optimizer_config.optimizer_name = "rangerlite"
    assert not train._use_sharding(args, "cuda", 2)


def test_resume_keeps_checkpoint_mode(tmp_path):
    ddp = tmp_path / "ddp.ckpt"
    sharded = tmp_path / "sharded.ckpt"
    torch.save({"state_dict": {}}, ddp)
    torch.save({"format": "nnue-sharded-v1", "state_dict": {}}, sharded)
    assert not train._use_sharding(_args(resume_from_checkpoint=str(ddp)), "cuda", 2)
    assert train._use_sharding(_args(resume_from_checkpoint=str(sharded)), "cuda", 2)
    with pytest.raises(ValueError, match="saved by sharded training"):
        train._use_sharding(
            _args(multi_gpu="ddp", resume_from_checkpoint=str(sharded)), "cuda", 2
        )
    with pytest.raises(ValueError, match="saved by DDP"):
        train._use_sharding(
            _args(multi_gpu="sharded", resume_from_checkpoint=str(ddp)), "cuda", 2
        )

"""Gathering FT/L1 shards must reproduce the full-model weights exactly."""

import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_sharded_ft import _Feature

import model.model as model_module
import model.sharded_model as sharded_model_module
from model.config import ModelConfig, NNUEConfig
from model.model import NNUEModel
from model.nnue import NNUE
from model.optimizers import OptimizerConfig


def _config():
    return NNUEConfig(
        features="Test",
        model_config=ModelConfig(
            L1=16, L2=4, L3=3, ls_keys="queen", ls_factors="queen", ls_remap_table="3:1"
        ),
        optimizer_config=OptimizerConfig(optimizer_name="adamw", adamw_fused=False),
    )


def _worker(rank, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    try:
        model_module.get_feature_cls = lambda _: [_Feature]
        sharded_model_module.get_feature_cls = lambda _: [_Feature]
        config = _config()
        torch.manual_seed(42)
        full = NNUE(config)
        sharded = NNUE(config, sharding=(rank, 2))
        sharded.model.load_unsharded(full.model)

        def collect():
            merged = NNUEModel("Test", config.model_config) if rank == 0 else None
            sharded.model.collect_unsharded_into(merged)
            return merged

        merged = collect()
        if rank == 0:
            expected = full.model.state_dict()
            got = merged.state_dict()
            assert expected.keys() == got.keys()
            for name, value in expected.items():
                torch.testing.assert_close(got[name], value, rtol=0, atol=0)

        # Different per-rank shard values must land in that rank's paired
        # columns of the full weight, not in rank order of the gathered list.
        layout = sharded.model.input.layout
        sharded_ids = {id(p) for p in sharded.model.sharded_parameters()}
        with torch.no_grad():
            for name, param in sharded.model.named_parameters():
                if id(param) in sharded_ids:
                    param.copy_(
                        torch.full_like(param, float(rank + 1))
                        + torch.arange(param.numel()).reshape(param.shape) * 1e-4
                    )
        merged = collect()
        if rank == 0:
            full_params = dict(merged.named_parameters())
            for name, param in sharded.model.named_parameters():
                if id(param) not in sharded_ids:
                    continue
                first, second = layout.columns
                width = layout.pair_width
                torch.testing.assert_close(
                    full_params[name][..., first], param[..., :width], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    full_params[name][..., second], param[..., width:], rtol=0, atol=0
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="gloo required",
)
def test_collect_unsharded_reproduces_full_model():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=2, join=True)

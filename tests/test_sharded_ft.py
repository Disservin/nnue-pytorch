"""Paired-column FT shards reproduce the full transform and its gradients."""

import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from data_loader.distributed_features import gather_packed_features
from model.modules.feature_transformer import (
    ComposedFeatureTransformer,
    FTShardLayout,
    ShardedFeatureTransformer,
)
from model.modules.feature_transformer.fused_ft_functions import _HAS_CUPY_KERNELS
from model.modules.features.input_feature import InputFeature
from model.modules.features.k32h3 import K32H3
from model.modules.features.pp_3wide import PP3Wide
from model.quantize import QuantizationConfig, QuantizationManager


class _Feature(InputFeature):
    FEATURE_NAME = INPUT_FEATURE_NAME = "Test"
    HASH = 42
    NUM_INPUTS = NUM_REAL_FEATURES = 16
    MAX_ACTIVE_FEATURES = 4

    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(16, width) * 0.1)
        self.virtual_weight = nn.Parameter(torch.randn(4, width) * 0.05)

    def merged_weight(self):
        return self.weight + self.virtual_weight.repeat(4, 1)

    def init_weights(self):
        pass

    def zero_virtual_weights(self):
        pass

    def coalesce(self):
        pass

    def get_export_weights(self):
        return self.merged_weight()

    def load_export_weights(self, weight):
        raise NotImplementedError


def _local_batch(rank, n):
    us = ((torch.arange(n) + rank) % 2).float().reshape(-1, 1)
    white = torch.full((n, 4), -1, dtype=torch.int32)
    black = torch.full_like(white, -1)
    for i in range(n):
        white[i, :2] = torch.tensor([(rank + i) % 16, (rank + i + 2) % 16])
        black[i, :3] = torch.tensor([(rank + i + j + 4) % 16 for j in range(3)])
    target = torch.zeros(n, 1)
    return us, 1 - us, white, black, target, target, torch.zeros(n, dtype=torch.int64)


def _compare_shard(
    rank, world, inputs, qat, feature_classes=None, *, device="cpu", backend="torch"
):
    torch.manual_seed(92)
    quantization = QuantizationManager(QuantizationConfig())
    feature_classes = feature_classes or [_Feature]
    full = ComposedFeatureTransformer(feature_classes, 16, quantization).to(device)
    with torch.no_grad():
        full.bias.uniform_(0.2, 0.4)
    shard = ShardedFeatureTransformer(
        feature_classes, 16, quantization, rank=rank, world_size=world
    ).to(device)
    shard.load_unsharded(full)
    args = (*(value.to(device) for value in inputs), qat, qat)
    expected = full(*args, backend="torch")
    actual = shard(*args, backend=backend)
    torch.testing.assert_close(
        actual, shard.layout.select_columns(expected), atol=1e-4, rtol=1e-4
    )
    # Different gradients per full output column catch both perspective and
    # paired-half routing errors; both losses use global-batch normalization.
    upstream = (
        torch.arange(expected.numel(), dtype=torch.float32, device=device).reshape_as(
            expected
        )
        / 100
    )
    (expected * upstream).sum().div(expected.shape[0]).backward()
    (actual * shard.layout.select_columns(upstream)).sum().div(
        expected.shape[0]
    ).backward()
    full_params = dict(full.named_parameters())
    for name, param in shard.named_parameters():
        torch.testing.assert_close(
            param.grad,
            shard.layout.select_columns(full_params[name].grad),
            atol=1e-4,
            rtol=1e-4,
        )
        # Check one SGD update, including virtual weights and bias.
        updated = param - 0.01 * param.grad
        reference = full_params[name] - 0.01 * full_params[name].grad
        torch.testing.assert_close(updated, shard.layout.select_columns(reference))


@pytest.mark.parametrize("qat", [False, True])
def test_all_shards_against_unsharded(qat):
    batch = _local_batch(1, 5)
    for rank in range(4):
        _compare_shard(rank, 4, batch[:4], qat)


def test_layout_validation():
    for width, world, rank in [(15, 4, 0), (4, 4, 0), (16, 0, 0), (16, 4, 4)]:
        with pytest.raises(ValueError):
            FTShardLayout(width, world, rank)


def test_initialization_is_reproducible_and_rank_distinct():
    def make(rank):
        torch.manual_seed(712)
        shard = ShardedFeatureTransformer(
            [_Feature],
            16,
            QuantizationManager(QuantizationConfig()),
            rank=rank,
            world_size=4,
        )
        return shard.features[0].weight.detach().clone(), torch.rand(1)

    zero, next_zero = make(0)
    one, next_one = make(1)
    again, next_again = make(0)
    assert not torch.equal(zero, one)
    torch.testing.assert_close(zero, again)
    torch.testing.assert_close(next_zero, next_one)
    torch.testing.assert_close(next_zero, next_again)


def test_shard_checkpoint_rejects_wrong_rank():
    quantization = QuantizationManager(QuantizationConfig())
    source = ShardedFeatureTransformer(
        [_Feature], 16, quantization, rank=0, world_size=4
    )
    matching = ShardedFeatureTransformer(
        [_Feature], 16, quantization, rank=0, world_size=4
    )
    matching.load_state_dict(source.state_dict())
    for name, param in source.named_parameters():
        torch.testing.assert_close(param, dict(matching.named_parameters())[name])
    other = ShardedFeatureTransformer(
        [_Feature], 16, quantization, rank=1, world_size=4
    )
    with pytest.raises(ValueError, match="layout"):
        other.load_state_dict(source.state_dict())


def test_real_feature_components():
    batch = list(_local_batch(0, 2))
    batch[2][:, 1] = K32H3.NUM_INPUTS + 3
    batch[3][:, 1] = K32H3.NUM_INPUTS + 4
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        _compare_shard(2, 4, batch[:4], True, [K32H3, PP3Wide])
    finally:
        torch.set_num_threads(threads)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _HAS_CUPY_KERNELS,
    reason="CUDA and CuPy are required",
)
@pytest.mark.parametrize("qat", [False, True])
def test_fused_cuda_shard(qat):
    _compare_shard(1, 4, _local_batch(1, 7)[:4], qat, device="cuda", backend="fused")


def _worker(rank, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank=rank, world_size=4, timeout=timedelta(seconds=30)
    )
    try:
        gathered = gather_packed_features(_local_batch(rank, 2))
        inputs = (
            gathered.us,
            gathered.them,
            gathered.white_indices,
            gathered.black_indices,
        )
        for qat in (False, True):
            _compare_shard(rank, 4, inputs, qat)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable"
)
def test_four_rank_global_batch_and_ft():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=4, join=True)

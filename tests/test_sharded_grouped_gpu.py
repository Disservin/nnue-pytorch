"""Two-GPU grouped-versus-dense parity for the sharded L1 path."""

import copy
import os
import socket
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from model.modules import stacked_linear
from model.modules.sharded_l1 import ShardedL1
from model.modules.stacked_linear import FactorizedStackedLinear
from model.quantize import QuantizationConfig, QuantizationManager
from model.sharded_backward import synchronize_replicated_gradients


def _check(rank, sizes, quantized, concentrated):
    torch.manual_seed(725)
    quant = QuantizationManager(QuantizationConfig())
    full = FactorizedStackedLinear(256, 16, 32, quant, "ls_l1")
    maps = [torch.arange(32) % 4]
    full.set_group_factors(maps)
    with torch.no_grad():
        for param in full.parameters():
            param.uniform_(-0.12, 0.12)
    shard = ShardedL1(256, 16, 32, quant, rank=rank, world_size=2, group_maps=maps)
    shard.load_unsharded(full)
    shard = shard.cuda(rank)
    other = copy.deepcopy(shard)
    n = sum(sizes)
    x = torch.randn(n, 128, device=rank)
    x[:, ::3] = 0
    x[0].zero_()
    indices = (
        torch.full((n,), 3, device=rank, dtype=torch.int64)
        if concentrated
        else torch.arange(n, device=rank) % 11
    )
    offsets = (0, sizes[0], n)
    upstream = torch.randn(n, 16, device=rank)
    results = []
    original = stacked_linear.grouped_l1
    for mode, layer in (("0", shard), ("1", other)):
        os.environ["NNUE_GROUPED_L1"] = mode
        inputs = x.detach().clone().requires_grad_(True)
        with patch.object(stacked_linear, "grouped_l1", wraps=original) as kernel:
            output = layer(inputs, indices, offsets, quantized)
            if mode == "1":
                kernel.assert_called_once()
            else:
                kernel.assert_not_called()
            (output * upstream[offsets[rank] : offsets[rank + 1]]).sum().div(
                n
            ).backward()
        synchronize_replicated_gradients(layer.replicated_bias_parameters())
        results.append(
            (
                output.detach(),
                inputs.grad.detach().clone(),
                [p.grad.detach().clone() for p in layer.parameters()],
            )
        )
    torch.testing.assert_close(results[0][0], results[1][0], atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(results[0][1], results[1][1], atol=2e-5, rtol=2e-4)
    for left, right in zip(results[0][2], results[1][2]):
        torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-4)
    for layer in (shard, other):
        torch.optim.AdamW(
            layer.parameters(), lr=0.0008, weight_decay=0.02, fused=False
        ).step()
    for left, right in zip(shard.parameters(), other.parameters()):
        torch.testing.assert_close(left, right, atol=3e-5, rtol=3e-4)


def _worker(rank, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    dist.init_process_group(
        "nccl", rank=rank, world_size=2, timeout=timedelta(seconds=120)
    )
    try:
        if stacked_linear.grouped_l1 is None:
            raise RuntimeError("grouped CUDA dependencies are required for this test")
        for sizes in ([0, 9], [4, 5]):
            for quantized in (False, True):
                for concentrated in (False, True):
                    _check(rank, sizes, quantized, concentrated)
        # 288 / 2 = 144 columns, outside the full-warp kernel contract.
        quant = QuantizationManager(QuantizationConfig())
        fallback = ShardedL1(288, 16, 32, quant, rank=rank, world_size=2).cuda(rank)
        x = torch.randn(5, 144, device=rank)
        indices = torch.arange(5, device=rank)
        os.environ["NNUE_GROUPED_L1"] = "1"
        with patch.object(
            stacked_linear, "grouped_l1", wraps=stacked_linear.grouped_l1
        ) as kernel:
            fallback(x, indices, (0, 2, 5))
            kernel.assert_not_called()
        autocast_layer = ShardedL1(256, 16, 32, quant, rank=rank, world_size=2).cuda(
            rank
        )
        with (
            patch.object(
                stacked_linear, "grouped_l1", wraps=stacked_linear.grouped_l1
            ) as kernel,
            torch.autocast("cuda"),
            torch.no_grad(),
        ):
            autocast_layer(x[:, :128], indices, (0, 2, 5))
            kernel.assert_not_called()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available()
    or not dist.is_nccl_available()
    or torch.cuda.device_count() < 2
    or torch.version.hip is not None
    or stacked_linear.grouped_l1 is None
    or any(
        torch.cuda.get_device_capability(i)[0] < 8
        for i in range(min(2, torch.cuda.device_count()))
    ),
    reason="Two NVIDIA SM80+ GPUs, NCCL, CuPy and Triton required",
)
def test_grouped_sharded_l1():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=2, join=True)

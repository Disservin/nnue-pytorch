"""Grouped-versus-dense parity for sharded L1 and the replicated tail."""

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
from model.modules.stacked_linear import FactorizedStackedLinear, StackedLinear
from model.quantize import QuantizationConfig, QuantizationManager


def _check(rank, local_size, quantized, concentrated):
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
    n = 2 * local_size
    x = torch.randn(n, 128, device=rank)
    x[:, ::3] = 0
    x[0].zero_()
    indices = (
        torch.full((n,), 3, device=rank, dtype=torch.int64)
        if concentrated
        else torch.arange(n, device=rank) % 11
    )
    upstream = torch.randn(n, 16, device=rank)
    results = []
    original = stacked_linear.grouped_l1
    for mode, layer in (("0", shard), ("1", other)):
        inputs = x.detach().clone().requires_grad_(True)
        with (
            patch.object(stacked_linear, "grouped_l1_preferred", return_value=mode == "1"),
            patch.object(stacked_linear, "grouped_l1", wraps=original) as kernel,
        ):
            output = layer(inputs, indices, quantized)
            if mode == "1":
                kernel.assert_called_once()
            else:
                kernel.assert_not_called()
            (output * upstream.chunk(2)[rank]).sum().div(n).backward()
        for p in layer.replicated_bias_parameters():
            dist.all_reduce(p.grad)
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
        for local_size in (1, 4, 9):
            for quantized in (False, True):
                for concentrated in (False, True):
                    _check(rank, local_size, quantized, concentrated)
        # 288 / 2 = 144 columns, outside the full-warp kernel contract.
        quant = QuantizationManager(QuantizationConfig())
        fallback = ShardedL1(288, 16, 32, quant, rank=rank, world_size=2).cuda(rank)
        x = torch.randn(4, 144, device=rank)
        indices = torch.arange(4, device=rank)
        with patch.object(
            stacked_linear, "grouped_l1", wraps=stacked_linear.grouped_l1
        ) as kernel:
            fallback(x, indices)
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
            autocast_layer(x[:, :128], indices)
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


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or stacked_linear.grouped_l1 is None
    or torch.cuda.get_device_capability()[0] < 8,
    reason="NVIDIA SM80+, CuPy and Triton required",
)
@pytest.mark.parametrize(
    "width,outputs,layer_key", [(64, 32, "ls_l2"), (128, 1, "ls_output")],
)
def test_grouped_tail_matches_dense(width, outputs, layer_key):
    torch.manual_seed(725)
    layer = StackedLinear(width, outputs, 92, QuantizationManager(QuantizationConfig()), layer_key).cuda()
    with torch.no_grad():
        layer.linear.weight.uniform_(-0.12, 0.12)
        layer.linear.bias.uniform_(-0.12, 0.12)
    inputs = torch.randn(257, width, device="cuda", requires_grad=True)
    indices = torch.arange(257, device="cuda") % 87
    upstream = torch.randn(257, outputs, device="cuda")
    parameters = (inputs, *layer.parameters())

    with patch.object(stacked_linear, "grouped_l1_preferred", return_value=False):
        expected = layer(inputs, indices, fake_quantize_weights=True)
    with patch.object(stacked_linear, "grouped_l1", wraps=stacked_linear.grouped_l1) as kernel:
        actual = layer(inputs, indices, fake_quantize_weights=True)
        kernel.assert_called_once()

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    for grad, ref in zip(
        torch.autograd.grad(actual, parameters, upstream),
        torch.autograd.grad(expected, parameters, upstream),
    ):
        torch.testing.assert_close(grad, ref, atol=3e-5, rtol=3e-4)

"""Compare FT/L1 tensor parallelism with the complete unsharded model."""

import copy
import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_sharded_ft import _Feature, _local_batch

from data_loader.distributed_features import gather_packed_features
from model.config import LossParams
from model.modules.config import LayerStacksConfig
from model.modules.feature_transformer import (
    ComposedFeatureTransformer,
    ShardedFeatureTransformer,
)
from model.modules.layer_stacks import LayerStacks
from model.modules.sharded_l1 import ShardedL1, sum_reduce_scatter_batch
from model.nnue import calculate_sf_loss, calculate_sf_loss_components
from model.quantize import QuantizationConfig, QuantizationManager
from model.sharded_backward import (
    backward_weighted_loss,
    clip_sharded_grad_norm_,
    global_weighted_loss,
    synchronize_replicated_gradients,
)


def _make_reference():
    torch.manual_seed(419)
    quant = QuantizationManager(QuantizationConfig())
    ft = ComposedFeatureTransformer([_Feature], 16, quant)
    config = LayerStacksConfig(
        L1=16, L2=4, L3=3, ls_keys="queen", ls_factors="queen", ls_remap_table="3:1"
    )
    layers = LayerStacks(4, config, quant)
    with torch.no_grad():
        ft.bias.uniform_(0.25, 0.4)
        # Exercise nonzero base, virtual and group contributions for every layer.
        for name, param in layers.named_parameters():
            if "group_factor" in name or "factorized" in name:
                param.uniform_(-0.09, 0.09)
    return quant, ft, layers


def _check(rank, qat_acts, qat_weights):
    sizes = [2, 0, 3, 1]
    local_batch = list(_local_batch(rank, sizes[rank]))
    local_batch[-1] = (torch.arange(sizes[rank], dtype=torch.int64) + rank) % 4
    batch = gather_packed_features(tuple(local_batch))
    quant, full_ft, full_layers = _make_reference()
    tail = copy.deepcopy(full_layers)
    shard_ft = ShardedFeatureTransformer([_Feature], 16, quant, rank=rank, world_size=4)
    shard_ft.load_unsharded(full_ft)
    maps = [
        getattr(full_layers.l1, f"group_map_{i}")
        for i in range(len(full_layers.l1.group_maps))
    ]
    shard_l1 = ShardedL1(
        16,
        full_layers.L2,
        full_layers.count,
        quant,
        rank=rank,
        world_size=4,
        group_maps=maps,
    )
    shard_l1.load_unsharded(full_layers.l1)
    inputs = (batch.us, batch.them, batch.white_indices, batch.black_indices)
    indices = full_layers.stack_remap[batch.layer_stack_indices]
    local_indices = indices[batch.local_slice]

    full_ft_out = full_ft(*inputs, qat_acts, qat_weights, backend="torch")
    full_l1 = full_layers.l1(full_ft_out, indices, qat_weights)
    expected = full_layers.forward_from_l1(full_l1, indices, qat_acts, qat_weights)
    shard_ft_out = shard_ft(*inputs, qat_acts, qat_weights, backend="torch")
    local_l1 = shard_l1(shard_ft_out, indices, batch.offsets, qat_weights)
    actual = tail.forward_from_l1(local_l1, local_indices, qat_acts, qat_weights)
    torch.testing.assert_close(
        local_l1, full_l1[batch.local_slice], atol=2e-6, rtol=1e-5
    )
    torch.testing.assert_close(
        actual, expected[batch.local_slice], atol=2e-6, rtol=1e-5
    )
    score = torch.linspace(-1600, 1600, len(indices)).reshape(-1, 1)
    outcome = (torch.arange(len(indices)).reshape(-1, 1) % 3).float()
    loss_params = LossParams(w1=10.0, w2=0.5, qp_asymmetry=0.25)
    reference_loss = calculate_sf_loss(expected * 600, score, outcome, loss_params, 0.7)
    numerator, weight = calculate_sf_loss_components(
        actual * 600,
        score[batch.local_slice],
        outcome[batch.local_slice],
        loss_params,
        0.7,
    )
    shard_params = list(shard_ft.parameters()) + shard_l1.sharded_parameters()
    tail_params = [
        p for name, p in tail.named_parameters() if not name.startswith("l1.")
    ]
    replicated = shard_l1.replicated_bias_parameters() + tail_params
    reference_loss.backward()
    backward_weighted_loss(numerator, weight, replicated)

    def check_param(actual_param, expected_param, *, columns=False):
        grad = actual_param.grad
        assert grad is not None
        ref = expected_param.grad
        if columns:
            ref = shard_ft.layout.select_columns(ref)
        torch.testing.assert_close(grad, ref, atol=3e-6, rtol=2e-5)
        expected_data = (
            shard_ft.layout.select_columns(expected_param)
            if columns
            else expected_param
        )
        torch.testing.assert_close(
            actual_param - 0.01 * grad, expected_data - 0.01 * ref, atol=3e-6, rtol=2e-5
        )

    full_params = dict(full_ft.named_parameters())
    for name, param in shard_ft.named_parameters():
        check_param(param, full_params[name], columns=True)
    full_params = dict(full_layers.l1.named_parameters())
    for name, param in shard_l1.named_parameters():
        check_param(param, full_params[name], columns=param.ndim == 2)
    full_params = dict(full_layers.named_parameters())
    for name, param in tail.named_parameters():
        if not name.startswith("l1."):
            check_param(param, full_params[name])

    reference_params = list(full_ft.parameters()) + list(full_layers.parameters())
    reference_norm = torch.nn.utils.clip_grad_norm_(reference_params, 0.01)
    actual_norm = clip_sharded_grad_norm_(shard_params, replicated, 0.01)
    torch.testing.assert_close(
        actual_norm.float(), reference_norm, atol=3e-6, rtol=2e-5
    )
    optimizer = torch.optim.SGD(shard_params + replicated, lr=0.01)
    reference_optimizer = torch.optim.SGD(reference_params, lr=0.01)
    optimizer.step()
    reference_optimizer.step()
    full_params = dict(full_ft.named_parameters())
    for name, param in shard_ft.named_parameters():
        torch.testing.assert_close(
            param,
            shard_ft.layout.select_columns(full_params[name]),
            atol=3e-6,
            rtol=2e-5,
        )
    full_params = dict(full_layers.l1.named_parameters())
    for name, param in shard_l1.named_parameters():
        reference = (
            shard_ft.layout.select_columns(full_params[name])
            if param.ndim == 2
            else full_params[name]
        )
        torch.testing.assert_close(param, reference, atol=3e-6, rtol=2e-5)
    full_params = dict(full_layers.named_parameters())
    for name, param in tail.named_parameters():
        if not name.startswith("l1."):
            torch.testing.assert_close(param, full_params[name], atol=3e-6, rtol=2e-5)


def _worker(rank, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank=rank, world_size=4, timeout=timedelta(seconds=45)
    )
    try:
        for acts, weights in (
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        ):
            _check(rank, acts, weights)
        used = torch.nn.Parameter(torch.tensor([1.0]))
        unused = torch.nn.Parameter(torch.tensor([2.0]))
        used.grad = torch.tensor([float(rank + 1)]) if rank in (0, 2) else None
        synchronize_replicated_gradients([used, unused])
        torch.testing.assert_close(used.grad, torch.tensor([4.0]))
        assert unused.grad is None
        torch.optim.AdamW([used, unused], lr=0.1, weight_decay=0.1).step()
        torch.testing.assert_close(unused, torch.tensor([2.0]))
        with pytest.raises(ValueError, match="finite and positive"):
            global_weighted_loss(
                torch.tensor(0.0, requires_grad=True), torch.tensor(0.0)
            )
        sparse_param = torch.nn.Parameter(torch.zeros(2))
        if rank == 1:
            sparse_param.grad = torch.sparse_coo_tensor(
                torch.tensor([[0]]), torch.tensor([1.0]), (2,)
            )
        with pytest.raises(ValueError, match="sparse"):
            synchronize_replicated_gradients([sparse_param])
        empty = torch.empty(0, 4, requires_grad=True)
        empty_local = sum_reduce_scatter_batch(empty, (0, 0, 0, 0, 0))
        empty_local.sum().backward()
        assert empty.grad.shape == (0, 4)

        even = dist.new_group([0, 2])
        odd = dist.new_group([1, 3])
        group = even if rank % 2 == 0 else odd
        group_rank = dist.get_rank(group)
        value = torch.full((3, 2), float(rank + 1), requires_grad=True)
        result = sum_reduce_scatter_batch(value, (0, 1, 3), group)
        expected_sum = 4.0 if rank % 2 == 0 else 6.0
        torch.testing.assert_close(result, torch.full_like(result, expected_sum))
        (result * (group_rank + 1)).sum().backward()
        expected_grad = torch.tensor([[1.0, 1.0], [2.0, 2.0], [2.0, 2.0]])
        torch.testing.assert_close(value.grad, expected_grad)
        sharded = torch.nn.Parameter(torch.tensor([0.0]))
        sharded.grad = torch.tensor([float(rank + 1)])
        replica = torch.nn.Parameter(torch.tensor([0.0]))
        replica.grad = torch.tensor([float(rank + 1)])
        replica64 = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float64))
        replica64.grad = torch.tensor([1.0], dtype=torch.float64)
        synchronize_replicated_gradients([replica, replica64], group)
        torch.testing.assert_close(replica.grad, torch.tensor([expected_sum]))
        torch.testing.assert_close(
            replica64.grad, torch.tensor([2.0], dtype=torch.float64)
        )
        norm = clip_sharded_grad_norm_([sharded], [replica, replica64], 100, group)
        expected_norm_squared = 30.0 if rank % 2 == 0 else 60.0
        torch.testing.assert_close(
            norm, torch.tensor(expected_norm_squared, dtype=torch.float64).sqrt()
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable"
)
def test_four_rank_forward_backward_and_step():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=4, join=True)


def _nccl_worker(rank, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", rank=rank, world_size=4, timeout=timedelta(seconds=45)
    )
    try:
        # Exercise equal splits with a non-contiguous source, then the uneven
        # and empty-rank cases below.
        source = torch.full((4, 8), float(rank + 1), device="cuda").t()
        source.requires_grad_()
        equal = sum_reduce_scatter_batch(source, (0, 2, 4, 6, 8))
        torch.testing.assert_close(equal, torch.full_like(equal, 10))
        (equal * (rank + 1)).sum().backward()
        expected_equal = torch.arange(1, 5, device="cuda").repeat_interleave(2)
        torch.testing.assert_close(source.grad, expected_equal[:, None].expand(8, 4).float())
        offsets = (0, 2, 2, 5, 6)
        value = torch.full((6, 4), float(rank + 1), device="cuda", requires_grad=True)
        local = sum_reduce_scatter_batch(value, offsets)
        torch.testing.assert_close(local, torch.full_like(local, 10))
        (local * (rank + 1)).sum().backward()
        expected = (
            torch.tensor([1, 1, 3, 3, 3, 4], device="cuda").reshape(6, 1).expand(6, 4)
        )
        torch.testing.assert_close(value.grad, expected.to(torch.float32))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available()
    or not dist.is_nccl_available()
    or torch.cuda.device_count() < 4,
    reason="Four CUDA GPUs and NCCL required",
)
def test_nccl_reduce_scatter_and_backward():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_nccl_worker, args=(port,), nprocs=4, join=True)


def test_single_rank_reduce_scatter():
    x = torch.randn(3, 5, requires_grad=True)
    result = sum_reduce_scatter_batch(x, (0, 3))
    torch.testing.assert_close(result, x)
    result.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))
    with pytest.raises(ValueError):
        sum_reduce_scatter_batch(x, (0, 2))


def test_rank_initialization():
    quant = QuantizationManager(QuantizationConfig())
    torch.manual_seed(42)
    first = ShardedL1(16, 4, 3, quant, rank=0, world_size=4)
    torch.manual_seed(42)
    second = ShardedL1(16, 4, 3, quant, rank=1, world_size=4)
    assert not torch.equal(first.linear.weight, second.linear.weight)
    torch.testing.assert_close(first.linear.bias, second.linear.bias)


def test_checkpoint_checks_group_mapping():
    quant = QuantizationManager(QuantizationConfig())
    source = ShardedL1(
        16, 4, 3, quant, rank=0, world_size=4, group_maps=[torch.tensor([0, 1, 0])]
    )
    same = ShardedL1(
        16, 4, 3, quant, rank=0, world_size=4, group_maps=[torch.tensor([0, 1, 0])]
    )
    same.load_state_dict(source.state_dict())
    different = ShardedL1(
        16, 4, 3, quant, rank=0, world_size=4, group_maps=[torch.tensor([1, 0, 0])]
    )
    with pytest.raises(ValueError, match="layout"):
        different.load_state_dict(source.state_dict())

"""Backward utilities for a sharded FT/L1 with a replicated local-batch tail.

All ranks must supply replicated parameters in the same order and call the
collectives in the same order. These helpers do not wrap parameters in DDP or
synchronize their initial values.
"""

from collections import OrderedDict
from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch import nn


def _world(group):
    return (
        dist.get_world_size(group)
        if dist.is_available() and dist.is_initialized()
        else 1
    )


def global_weighted_loss(
    local_numerator: torch.Tensor, local_weight: torch.Tensor, group=None
):
    """Return this rank's contribution to the global weighted mean loss.

    The denominator is target-derived and must not require gradients. An empty
    rank must still pass a graph-connected zero numerator so backward enters
    the L1 all-gather. The all-empty case is rejected on every rank.
    """
    if local_numerator.ndim or local_weight.ndim or local_weight.requires_grad:
        raise ValueError("numerator and non-differentiable weight must be scalars")
    denominator = local_weight.detach().clone()
    if _world(group) > 1:
        dist.all_reduce(denominator, op=dist.ReduceOp.SUM, group=group)
    if not bool(torch.isfinite(denominator) & (denominator > 0)):
        raise ValueError("global loss weight must be finite and positive")
    return local_numerator / denominator


def backward_weighted_loss(
    local_numerator, local_weight, replicated_parameters, group=None
):
    """Backpropagate the local weighted contribution, then sum replica gradients.

    Start with cleared gradients and call exactly once per optimizer step on
    every rank. Returns the detached local contribution, not the global loss.
    """
    loss = global_weighted_loss(local_numerator, local_weight, group)
    loss.backward()
    synchronize_replicated_gradients(replicated_parameters, group)
    return loss.detach()


@torch.no_grad()
def synchronize_replicated_gradients(parameters: Iterable[nn.Parameter], group=None):
    """SUM replicated gradients, preserving None for globally unused parameters.

    Ranks with a locally absent gradient contribute zero when another rank used
    that parameter. Parameters must be ordered identically across ranks. Only
    dense gradients on the group's collective device are supported.
    """
    params = list(parameters)
    if len({id(p) for p in params}) != len(params):
        raise ValueError("replicated parameters must not contain duplicates")
    if not params:
        return
    if any(p.device != params[0].device for p in params):
        raise ValueError("replicated parameters must be on one device")
    presence = torch.tensor(
        [
            0 if p.grad is None else 2 if p.grad.layout != torch.strided else 1
            for p in params
        ],
        dtype=torch.int32,
        device=params[0].device,
    )
    if _world(group) > 1:
        dist.all_reduce(presence, op=dist.ReduceOp.MAX, group=group)
    statuses = presence.tolist()
    if 2 in statuses:
        raise ValueError("sparse replicated gradients are not supported")

    # One collective per dtype, rather than one per parameter. Do not include
    # globally unused parameters: AdamW must not apply decay to them.
    buckets = OrderedDict()
    for p, status in zip(params, statuses):
        if status:
            buckets.setdefault(p.dtype, []).append(p)
    for bucket in buckets.values():
        flat = torch.cat(
            [
                (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                for p in bucket
            ]
        )
        if _world(group) > 1:
            dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)
        offset = 0
        for p in bucket:
            grad = flat[offset : offset + p.numel()].view_as(p)
            if p.grad is None:
                p.grad = grad.clone()
            else:
                p.grad.copy_(grad)
            offset += p.numel()


@torch.no_grad()
def clip_sharded_grad_norm_(
    sharded_parameters, replicated_parameters, max_norm, group=None
):
    """Clip using the full-model L2 norm, counting replicated gradients once.

    Call after synchronizing replicated gradients. Sharded parameters must
    represent disjoint pieces of the full model. Returns the unclipped norm.
    """
    sharded, replicated = list(sharded_parameters), list(replicated_parameters)
    params = sharded + replicated
    if not params:
        raise ValueError("at least one parameter is required")
    if len({id(p) for p in params}) != len(params):
        raise ValueError("parameter sets must be disjoint and contain no duplicates")
    if any(p.device != params[0].device for p in params):
        raise ValueError("all parameters must be on one device")
    if max_norm < 0:
        raise ValueError("max_norm must be nonnegative")
    # Accumulate the per-parameter norms in float64 without copying full
    # feature-transformer gradients into float64 buffers.
    total = torch.zeros((), dtype=torch.float64, device=params[0].device)
    for p in sharded:
        if p.grad is not None:
            total += (
                torch.linalg.vector_norm(p.grad.detach()).to(torch.float64).square()
            )
    if _world(group) > 1:
        dist.all_reduce(total, op=dist.ReduceOp.SUM, group=group)
    for p in replicated:
        if p.grad is not None:
            total += (
                torch.linalg.vector_norm(p.grad.detach()).to(torch.float64).square()
            )
    norm = total.sqrt()
    scale = (max_norm / (norm + 1e-6)).clamp(max=1.0)
    for p in params:
        if p.grad is not None:
            p.grad.mul_(scale.to(dtype=p.grad.dtype))
    return norm

"""Backward and gradient clipping for the sharded FT/L1 with a replicated tail.

All ranks must pass the replicated parameters in the same order and call
these functions in the same order.
"""

import torch
import torch.distributed as dist


def _world(group):
    return (
        dist.get_world_size(group)
        if dist.is_available() and dist.is_initialized()
        else 1
    )


def backward_weighted_loss(
    local_numerator, local_weight, replicated_parameters, ft_boundary, group=None
):
    """Backpropagate this rank's share of the global weighted mean loss.

    The loss is the sum of every rank's numerator over the sum of every
    rank's target-derived weight. ``ft_boundary`` is ``(ft_output,
    l1_input)``, where the forward detached the FT output: the first
    backward stops there, then the summed replica gradients are reduced
    while the local FT backward runs. Start with cleared gradients. Returns
    the detached local contribution, not the global loss.
    """
    world = _world(group)
    denominator = local_weight.detach().clone()
    if world > 1:
        dist.all_reduce(denominator, group=group)
    loss = local_numerator / denominator
    loss.backward()

    flat = torch.cat([p.grad.reshape(-1) for p in replicated_parameters])
    work = dist.all_reduce(flat, group=group, async_op=True) if world > 1 else None
    ft_output, l1_input = ft_boundary
    ft_output.backward(l1_input.grad)
    if work is not None:
        work.wait()
    offset = 0
    for p in replicated_parameters:
        p.grad = flat[offset : offset + p.numel()].view_as(p)
        offset += p.numel()
    return loss.detach()


@torch.no_grad()
def clip_sharded_grad_norm_(sharded_parameters, replicated_parameters, max_norm, group=None):
    """Clip by the full model's gradient L2 norm; returns the unclipped norm.

    Shard norms are summed across the ranks; the already-reduced replica
    gradients are counted once.
    """
    sharded = [p.grad for p in sharded_parameters if p.grad is not None]
    replicated = [p.grad for p in replicated_parameters if p.grad is not None]

    def norm_squared(grads):
        if not grads:
            return torch.zeros((), dtype=torch.float64, device=(sharded + replicated)[0].device)
        return torch.stack(torch._foreach_norm(grads)).to(torch.float64).square().sum()

    total = norm_squared(sharded)
    if _world(group) > 1:
        dist.all_reduce(total, group=group)
    norm = (total + norm_squared(replicated)).sqrt()
    scale = (max_norm / (norm + 1e-6)).clamp(max=1.0).to(torch.float32)
    torch._foreach_mul_(sharded + replicated, scale)
    return norm

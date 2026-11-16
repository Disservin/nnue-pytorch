"""Benchmark-only batched implementation of global sharded gradient clipping."""
from collections import defaultdict

import torch
import torch.distributed as dist


@torch.no_grad()
def foreach_sharded_clip(sharded_parameters, replicated_parameters, max_norm, group=None):
    sharded = [p.grad for p in sharded_parameters if p.grad is not None]
    replicated = [p.grad for p in replicated_parameters if p.grad is not None]
    gradients = sharded + replicated
    if not gradients:
        raise ValueError("at least one gradient is required")
    device = gradients[0].device

    def squared_norm(grads):
        buckets = defaultdict(list)
        for grad in grads:
            buckets[grad.dtype].append(grad.detach())
        contributions = []
        for bucket in buckets.values():
            norms = torch._foreach_norm(bucket, 2)
            contributions.append(torch.stack(norms).to(torch.float64).square().sum())
        if not contributions:
            return torch.zeros((), dtype=torch.float64, device=device)
        return torch.stack(contributions).sum()

    total = squared_norm(sharded)
    if dist.is_initialized() and dist.get_world_size(group) > 1:
        dist.all_reduce(total, op=dist.ReduceOp.SUM, group=group)
    total += squared_norm(replicated)
    norm = total.sqrt()
    scale = (max_norm / (norm + 1e-6)).clamp(max=1.0)
    buckets = defaultdict(list)
    for grad in gradients:
        buckets[grad.dtype].append(grad)
    for dtype, bucket in buckets.items():
        torch._foreach_mul_(bucket, scale.to(dtype=dtype))
    return norm

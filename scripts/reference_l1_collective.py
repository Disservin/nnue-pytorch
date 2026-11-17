"""Original padded L1 collective retained for regression tests and benchmarks."""
import torch
import torch.distributed as dist


class ReferenceReduceScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, offsets, group):
        world = len(offsets) - 1
        rank = dist.get_rank(group) if world > 1 else 0
        sizes = tuple(offsets[i + 1] - offsets[i] for i in range(world))
        max_size = max(1, *sizes)
        blocks = values.new_zeros((world, max_size, values.shape[1]))
        for i, size in enumerate(sizes):
            blocks[i, :size].copy_(values[offsets[i]:offsets[i + 1]])
        result = values.new_empty((max_size, values.shape[1]))
        if world == 1:
            result.copy_(blocks[0])
        elif dist.get_backend(group) == "nccl":
            dist.reduce_scatter_tensor(result, blocks.flatten(0, 1), group=group)
        else:
            dist.all_reduce(blocks, group=group)
            result.copy_(blocks[rank])
        ctx.sizes, ctx.max_size, ctx.rank, ctx.group = sizes, max_size, rank, group
        return result[:sizes[rank]].clone()

    @staticmethod
    def backward(ctx, grad_output):
        padded = grad_output.new_zeros((ctx.max_size, grad_output.shape[1]))
        padded[:ctx.sizes[ctx.rank]].copy_(grad_output)
        blocks = [torch.empty_like(padded) for _ in ctx.sizes]
        if len(blocks) == 1:
            blocks[0] = padded
        else:
            dist.all_gather(blocks, padded, group=ctx.group)
        return torch.cat([b[:n] for b, n in zip(blocks, ctx.sizes)]), None, None

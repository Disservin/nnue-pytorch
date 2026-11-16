"""Measurement-only bucketed L1 bias-gradient experiment."""
import torch
import triton

from model.modules.grouped_linear import _bucketed_bias_gradient, _route_rows


class _BucketedBiasAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, bias, indices, count, out_features):
        ctx.save_for_backward(indices)
        ctx.count, ctx.out_features = count, out_features
        return values + bias.view(count, out_features)[indices]

    @staticmethod
    def backward(ctx, grad):
        (indices,) = ctx.saved_tensors
        count, n = ctx.count, ctx.out_features
        if not len(indices):
            return grad, grad.new_zeros(count * n), None, None, None
        indices = indices.contiguous()
        grad = grad.contiguous()
        rows, counts = _route_rows(indices, count)
        split = 8
        partial = torch.empty((count, split, n), dtype=grad.dtype, device=grad.device)
        _bucketed_bias_gradient[(count, split)](
            grad, rows, counts, partial, len(indices), n, split,
            max(16, triton.next_power_of_2(n)), num_warps=4,
        )
        return grad, partial.sum(dim=1).reshape(-1), None, None, None


def bucketed_bias_add(values, bias, indices, count, out_features):
    return _BucketedBiasAdd.apply(values, bias, indices, count, out_features)

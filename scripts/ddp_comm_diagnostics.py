"""Benchmark-only DDP hooks for measuring gradient communication exposure."""
import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook


def make_diagnostic_hook(model, mode, bucket_info):
    if mode not in ("hook", "skip-ft", "skip-all"):
        raise ValueError("invalid DDP diagnostic mode")
    names = {id(p): n.removeprefix("_orig_mod.") for n, p in model.named_parameters()}

    def diagnostic_hook(group, bucket):
        params = bucket.parameters()
        items = [(names[id(p)], p.numel() * p.element_size()) for p in params]
        ft_bytes = sum(size for name, size in items if name.startswith("input."))
        total_bytes = bucket.buffer().numel() * bucket.buffer().element_size()
        bucket_info[bucket.index()] = {
            "index": bucket.index(), "bytes": total_bytes, "ft_bytes": ft_bytes,
            "other_bytes": total_bytes - ft_bytes, "parameters": [name for name, _ in items],
        }
        if mode == "hook" or (mode == "skip-ft" and not ft_bytes):
            return allreduce_hook(group, bucket)
        if mode == "skip-ft" and total_bytes != ft_bytes:
            # Keep the non-FT part synchronized even if the same bucket also
            # holds FT gradients. Fail rather than silently dropping unrelated
            # gradients when the layout cannot be reduced as one slice.
            ranges, offset = [], 0
            for p, (name, _) in zip(params, items):
                end = offset + p.numel()
                if not name.startswith("input."):
                    if ranges and ranges[-1][1] == offset:
                        ranges[-1] = (ranges[-1][0], end)
                    else:
                        ranges.append((offset, end))
                offset = end
            if offset != bucket.buffer().numel() or len(ranges) != 1:
                raise RuntimeError("unsupported mixed DDP bucket layout in FT ablation")
            start, end = ranges[0]
            buffer = bucket.buffer()
            retained = buffer.narrow(0, start, end - start)
            retained.div_(dist.get_world_size(group))
            work = dist.all_reduce(retained, group=group, async_op=True)
            return work.get_future().then(lambda _: buffer)
        future = torch.futures.Future()
        future.set_result(bucket.buffer())
        return future

    return diagnostic_hook

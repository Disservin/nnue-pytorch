"""Compare the previous and current clipping loops on nettest parameter shapes.

Run with torchrun and --remap-cache pointing to the actual stack remap. The
gradients are synthetic and reused; these timings isolate clipping and are not
end-to-end training throughput measurements.
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.nnue import NNUE
from model.sharded_backward import clip_sharded_grad_norm_
from nettest_benchmark_settings import make_nettest_settings


@torch.no_grad()
def old_clip(sharded, replicated, max_norm):
    total = torch.zeros((), dtype=torch.float64, device=sharded[0].device)
    for p in sharded:
        if p.grad is not None:
            total += torch.linalg.vector_norm(p.grad.detach()).to(torch.float64).square()
    if dist.get_world_size() > 1:
        dist.all_reduce(total)
    for p in replicated:
        if p.grad is not None:
            total += torch.linalg.vector_norm(p.grad.detach()).to(torch.float64).square()
    norm = total.sqrt()
    scale = (max_norm / (norm + 1e-6)).clamp(max=1.0)
    for p in sharded + replicated:
        if p.grad is not None:
            p.grad.mul_(scale.to(dtype=p.grad.dtype))
    return norm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remap-cache", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0:
        parser.error("steps must be positive and warmup must be nonnegative")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    try:
        torch.set_num_threads(1)
        with open(args.remap_cache, encoding="utf-8") as remap_file:
            config, _, _ = make_nettest_settings(remap_file.read().strip())
        nnue = NNUE(config, max_epoch=560, num_batches_per_epoch=8192,
                    sharding=(dist.get_rank(), dist.get_world_size())).cuda(rank)
        sharded = nnue.model.sharded_parameters()
        replicated = nnue.model.replicated_parameters()
        for param in sharded + replicated:
            param.grad = torch.full_like(param, 0.001)
        for name, fn in (("old", old_clip), ("new", clip_sharded_grad_norm_),
                         ("new", clip_sharded_grad_norm_), ("old", old_clip)):
            for _ in range(args.warmup):
                fn(sharded, replicated, 2.0)
            torch.cuda.synchronize()
            dist.barrier()
            started = time.perf_counter()
            for _ in range(args.steps):
                fn(sharded, replicated, 2.0)
            torch.cuda.synchronize()
            elapsed = torch.tensor(time.perf_counter() - started, device=rank)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            if dist.get_rank() == 0:
                print(json.dumps({"case": name, "steps": args.steps,
                                  "ms_per_step": elapsed.item() * 1000 / args.steps,
                                  "sharded_tensors": len(sharded),
                                  "replicated_tensors": len(replicated)}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

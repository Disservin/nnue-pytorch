"""Measure isolated FT all-reduce, packed feature gather, and L1 exchange."""

import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist
from benchmark_sharded_adamw import make_batch

from data_loader.distributed_features import gather_packed_features
from model.modules.sharded_l1 import sum_reduce_scatter_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-batch", type=int, default=131072)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    if args.global_batch % world:
        raise ValueError("global batch must be divisible by the process count")
    torch.set_num_threads(4)
    device = torch.device("cuda", rank)
    batch = make_batch(args.global_batch // world, device, rank)
    # Three real feature tables, the K32H3 virtual table, and FT bias.
    ft_elements = (59808 + 4560 + 73728 + 768) * 1024 + 1024
    ft_gradient = torch.ones(ft_elements, dtype=torch.float32, device=device)
    partial_l1 = torch.randn(args.global_batch, 32, device=device, requires_grad=True)
    local_batch = args.global_batch // world
    offsets = tuple(i * local_batch for i in range(world + 1))

    def allreduce():
        dist.all_reduce(ft_gradient)

    def feature_gather():
        gather_packed_features(batch)

    def l1_exchange():
        partial_l1.grad = None
        local = sum_reduce_scatter_batch(partial_l1, offsets)
        local.sum().backward()

    results = {}
    for name, work in (
        ("ft_allreduce", allreduce),
        ("feature_gather", feature_gather),
        ("l1_reduce_scatter_and_backward", l1_exchange),
    ):
        values = []
        for iteration in range(args.warmup + args.steps):
            dist.barrier()
            torch.cuda.synchronize()
            start = time.perf_counter()
            work()
            torch.cuda.synchronize()
            elapsed = torch.tensor(time.perf_counter() - start, device=device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            if iteration >= args.warmup:
                values.append(elapsed.item())
        results[name] = {"seconds": values, "median_seconds": statistics.median(values)}
    if rank == 0:
        print(
            json.dumps(
                {
                    "world_size": world,
                    "global_batch": args.global_batch,
                    "ft_bytes": ft_elements * 4,
                    "active_per_perspective": 64,
                    "results": results,
                }
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

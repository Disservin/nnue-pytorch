"""Compare real-batch distributed feature reconstruction on multiple GPUs."""
import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_loader.distributed_features as features
from benchmark_sharded_adamw import pack_saved_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--batch", type=int, default=131072)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    torch.set_num_threads(4)
    path = os.path.join(args.batch_dir, f"batch-{rank}-{args.batch}.pt")
    batch = tuple(x.cuda(rank) for x in torch.load(path, map_location="cpu", weights_only=True))
    selector = features._use_fused_unpack
    features._use_fused_unpack = lambda device: False
    reference = features.gather_packed_features(batch)
    features._use_fused_unpack = selector
    actual = features.gather_packed_features(batch)
    for field in ("us", "them", "white_indices", "black_indices", "layer_stack_indices"):
        torch.testing.assert_close(getattr(actual, field), getattr(reference, field), rtol=0, atol=0)
    assert actual.offsets == reference.offsets
    native_batch = pack_saved_batch(batch)
    native_result = features.gather_packed_features(native_batch)
    for field in ("us", "them", "white_indices", "black_indices", "layer_stack_indices"):
        torch.testing.assert_close(getattr(native_result, field), getattr(reference, field), rtol=0, atol=0)
    assert native_result.offsets == reference.offsets
    del reference, actual, native_result
    results = {}
    for name in ("separate", "fused", "native", "native_repeat", "fused_repeat", "separate_repeat"):
        features._use_fused_unpack = (lambda device: False) if name.startswith("separate") else selector
        times = []
        for i in range(args.steps + 4):
            dist.barrier()
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = features.gather_packed_features(native_batch if name.startswith("native") else batch)
            torch.cuda.synchronize()
            elapsed = torch.tensor(time.perf_counter() - start, device=rank)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            if i >= 4:
                times.append(elapsed.item())
            del result
        results[name] = {"median_ms": statistics.median(times) * 1000,
                         "samples_ms": [t * 1000 for t in times]}
    if rank == 0:
        print(json.dumps({"gpu": torch.cuda.get_device_name(rank), "batch": args.batch,
                          "world": dist.get_world_size(), "results": results}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

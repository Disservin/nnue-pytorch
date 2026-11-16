"""Split saved two-rank benchmark batches without changing global order."""
import argparse
import os
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--batch", type=int, default=131072)
args = parser.parse_args()
os.makedirs(args.output, exist_ok=True)
for source_rank in range(2):
    path = os.path.join(args.input, f"batch-{source_rank}-{args.batch}.pt")
    batch = torch.load(path, map_location="cpu", weights_only=True)
    n = batch[0].shape[0]
    if n * 2 != args.batch or n % 2:
        raise ValueError("expected equally divided two-rank batch")
    for half in range(2):
        part = tuple(t[half * (n // 2):(half + 1) * (n // 2)].clone() for t in batch)
        target = os.path.join(args.output, f"batch-{2 * source_rank + half}-{args.batch}.pt")
        torch.save(part, target)
        print(target, flush=True)

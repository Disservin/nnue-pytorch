# Network training

For reproducible training and engine testing, follow the upstream guides:

- [Run a training workflow locally](https://github.com/vondele/nettest#local-execution).
- [Configure training stages and restarts](https://github.com/vondele/nettest#recipe-description).
- [Select trainer revisions, engines, and datasets](https://github.com/vondele/nettest#external-tools-and-data).
- [Run a workflow in CI](https://github.com/vondele/nettest#execution-in-the-ci-environment).

## Hardware and datasets

GPU training requires at least 16 GB of system RAM and 8 GB of GPU VRAM.
8 GB of VRAM is tight; more gives room for larger batches and networks.

The complete dataset set in the
[threats training recipe](https://github.com/vondele/nettest/blob/master/threats.yaml)
requires 800+ GB of storage, plus space for checkpoints, caches, and outputs.
Training can also use a single small binpack, which requires much less storage
but produces weaker networks. Use fast storage for data loading and allow
additional CPU and memory resources for engine testing.

## Multi-GPU training

The default multi-process mode wraps the model in DDP, which replicates the
feature transformer on every rank and all-reduces its full gradient. That
gradient is the largest tensor in the network; on hosts without NVLink it can
dominate step time and make two GPUs slower than one.

`--sharded` splits the feature transformer and L1 weights across the ranks
instead. Each rank stores one paired-column shard and the replicated tail; the
distributed forward exchanges L1 activations with a reduce-scatter/all-gather
pair, so the multi-hundred-MiB weight gradient is never reduced. Use it with
AdamW and at least two processes:

```
torchrun --standalone --nproc-per-node=2 ddp_launcher.py train.py \
    --sharded [the usual training options]
```

Checkpoints stay compatible with the rest of the tooling: rank 0 writes the
normal full-model checkpoint, and every rank additionally writes
`<checkpoint>.rankN` with its shard weights, AdamW moments, schedulers and RNG.
Resuming requires the same world size and every rank's state file; resuming
from a plain `.pt` model initialises the shards but starts a fresh optimizer.
SWA and non-CUDA backends are not supported in this mode.

See [Multi-GPU benchmark notes](benchmark_results.md) for measured throughput
on PCIe-only and NVLink hosts and for the communication accounting.

## Local trainer development

Use `run_docker.sh` to develop and test this checkout. See
[Developing and testing with Docker](development.md) for checks, the training
pipeline test, evaluation cross-checks, and logging.

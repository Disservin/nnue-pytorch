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

Launch one process per GPU with `torchrun`:

```
torchrun --standalone --nproc-per-node=2 ddp_launcher.py train.py [the usual training options]
```

By default (`--multi-gpu auto`) the feature transformer and L1 weights are
split across the GPUs. Each GPU stores one column shard of them plus a copy
of the small remaining layers, and the GPUs exchange the active features and
L1 activations of the batch instead of all-reducing the full FT gradient
(hundreds of MiB) every step. Without NVLink this makes multi-GPU training
several times faster than DDP; with NVLink both are about even, and sharding
uses less memory per GPU.

Sharding needs CUDA, the AdamW optimizer, no SWA, and an L1 size divisible by
twice the number of GPUs. When one of these does not hold, `auto` falls back
to DDP and prints why. `--multi-gpu sharded` makes that an error instead, and
`--multi-gpu ddp` always replicates the model.

Sharded checkpoints are full-model checkpoints written by rank 0, so the
conversion and evaluation tools read them as usual. Every rank also writes
`<checkpoint>.rankN` with its shard's AdamW state, schedulers and RNG.
Resuming from such a checkpoint keeps sharding and needs the same number of
GPUs and every rank file; a DDP checkpoint keeps resuming with DDP. Resuming
from a `.pt` model works in both modes, but sharding starts a fresh
optimizer.

See [Multi-GPU benchmark notes](benchmark_results.md) for measurements.

## Local trainer development

Use `run_docker.sh` to develop and test this checkout. See
[Developing and testing with Docker](development.md) for checks, the training
pipeline test, evaluation cross-checks, and logging.

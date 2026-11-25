# Sharded multi-GPU training

This is the default multi-GPU mode (`--multi-gpu auto`, see
[Training](training.md#multi-gpu-training)). This page describes how it
works and what it was measured to gain.

## Why

DDP keeps a full copy of the model on every GPU and all-reduces every
gradient after backward. For the threats feature set the feature transformer
(FT) is about 540 MiB of the network's ~590 MiB, and its gradient becomes
ready only at the very end of backward, so the all-reduce barely overlaps
with compute. On a PCIe host this dominates the step: two GPUs with DDP are
slower than one.

## How

With N GPUs, every rank owns a slice of `L1 / (2N)` paired output columns of
the FT, the matching input columns of L1, and a replica of the L1 biases,
L2 and output layers (`model/sharded_model.py`). One training step
(`trainer/sharded_adamw.py`):

1. All-gather the active feature IDs of every rank's local batch
   (`data_loader/distributed_features.py`), so each rank sees the global
   batch.
2. Run the FT shard on the global batch and multiply by the L1 column
   shard, giving every rank a partial L1 output for all positions.
3. Reduce-scatter the partial outputs: each rank gets the summed L1 output
   for its own local positions and runs the replicated tail on them.
4. In backward, all-gather the L1 output gradients so every rank can
   backpropagate its FT/L1 shard over the global batch. The FT gradient
   stays local; only the small replicated tail gradients are all-reduced.
5. Clip by the global gradient norm, counting replicated gradients once, and
   apply AdamW to the local shard and the replicas.

The per-step traffic is the packed feature IDs, one L1-sized reduce-scatter
and all-gather, and the tail gradients, instead of the whole FT gradient.
Per-GPU memory drops as well, since each GPU stores only its FT slice and
its AdamW moments.

Checkpoints: rank 0 gathers the shards into a normal full-model checkpoint,
and every rank writes `<checkpoint>.rankN` with its own optimizer,
scheduler and RNG state (`trainer/sharded_trainer.py`).

## Measurements

Global batch 131072, `Full_Threats+PP_3Wide+K32H3^`, FT width 1024, the
nettest recipe's layer stacks, AdamW, gradient and weight clipping. These
are throughput numbers only; neither mode was compared for convergence or
net strength beyond the parity tests in `tests/`.

Two A100-PCIE-40GB GPUs, PCIe 3.0 x16, no NVLink, `train.py` with the
60-file recipe mixture:

| Configuration | it/s | ms/step |
| --- | ---: | ---: |
| One A100 | 26.7 | 37 |
| Two A100, DDP | 10.0 | 100 |
| Two A100, sharded | 26.7 | 37 |

On this host two sharded GPUs match one GPU at the recipe batch size rather
than beating it: halving the FT/L1 work saves about as much time as the
feature and activation exchange costs. With a 262144 global batch, two
sharded GPUs reach 1.33x one GPU at that batch size.

Four H200 NVL GPUs with NVLink, 16 loader workers per rank:

| Path | Same saved batch repeated | Live loader |
| --- | ---: | ---: |
| Compiled DDP | 9.3–9.4M pos/s | 8.0–8.7M pos/s |
| Sharded | 10.5–10.6M pos/s | 8.1–8.7M pos/s |

With NVLink, sharding is faster on a fixed batch, but the live ranges
overlap: there, loader throughput and rank imbalance dominate the
variation. Peak allocated memory
was about 4.0 GiB per GPU for DDP and 1.9 GiB for sharding.

# Data loader performance

How a training batch gets from the binpacks to the GPU, where time goes at each stage, and which
options move the limits. All numbers are for the pretraining step's settings with `--l1 512`:
features `Full_Threats+PP_3Wide+K32H3^`, batch size 131072, the recipe's skip filters
(`random-fen-skipping 3`, `early-fen-skipping 18`, `soft-early-fen-skipping 32`,
`tree-sample-accept 0.12`), on an RTX 4090 (PCIe gen 3 x16), 32 CPUs, 31 GB RAM, binpacks on a
SATA SSD.

## The pipeline

1. **Readers** (C++, part of `--num-workers`) read binpack chunks, decode positions and apply the
   skip filters.
2. **Builders** (C++, the rest of `--num-workers`) turn the kept positions into a batch: feature
   indices in the packed layout, targets, layer-stack indices.
3. **Prefetch threads** (Python, `--data-loader-threads`) take finished batches, copy them into
   pinned memory, start the transfer to the GPU and run the model's prefetch hook (the tiled
   backward's per-batch preparation). Each has its own CUDA stream.
4. **The prefetch queue** (`--data-loader-queue-size`) holds batches that are ready on the GPU.
5. **The training step** takes a batch from the queue.

At the end of every epoch the trainer prints which stage limits training, for example:

```
Epoch  1 (Loader): step waited for data 0.3% of the epoch, loader idle (queue full) 38.1%,
in C++ fetch 0.5%, in transport 47.1% (per thread, 2 prefetch threads), ... ->
not loader bound (GPU step is the limit)
```

- `TRANSPORT BOUND`: raise `--data-loader-threads`.
- `DATA-LOADER BOUND`: the readers/builders are the limit; raise `--num-workers` if CPUs are free.
- `BURSTY`: raise `--data-loader-queue-size`.

## Stage capacities

| Stage | Capacity | Notes |
|---|---|---|
| GPU training step | ~42.7 it/s | the current limit |
| Readers + builders, 32 workers | ~51 batches/s | measured with no GPU work |
| Prefetch thread, each | ~40-43 batches/s | ~23 ms of host work per batch |
| PCIe transfer | ~250 batches/s | 13.1 GB/s measured, ~52 MB per batch |
| Disk (SATA SSD, ntfs3) | ~544 MB/s sequential | page-cache dependent |

### Readers and builders

The filters keep about 1 of every 27 positions read (5.38e9 kept out of 1.45e11 read over five
epochs), so a batch of 131072 positions means decoding ~3.5 million.

Builders are the more expensive side per position (~1.3 us per kept position for both
perspectives). With 1/4 of 32 workers as builders, the 8 builders ran at ~1.04 cores each and the
24 readers at ~0.82, so the builders limited the pipeline. The builder share at B >= 131072 is
now 5/16 (`FeaturedBatchStream::worker_thread_ratio`):

| Builders (32 workers) | Batches/s |
|---|---|
| 1/4 | 46.9 |
| 5/16 | 50.5-51.0 |
| 3/8 | 49.6 |
| 7/16 | 47.9 |

Fewer workers falls off quickly: training ran at 27.3 it/s with 24 workers, 42.6 with 28 and 42.7
with 32.

### Prefetch threads

Per batch, a prefetch thread spends:

| Work | Time |
|---|---|
| Copy the packed indices (48 MiB) into pinned memory | ~14 ms |
| Prefetch hook, mostly its host sync waiting on the GPU preparation | ~6 ms |
| Float block copy, transfer and unpack launches | ~3 ms |

The transfer to the GPU itself runs on the copy engine (~4 ms per batch) and overlaps with all of
this. The work does not depend on other batches and its slow parts release the GIL, so several
threads run it in parallel:

| `--data-loader-threads` | it/s | Step waited for data | Loader spare capacity | Peak GPU memory |
|---|---|---|---|---|
| 1 | 42.1 | 13-16% | ~2% | 7.7 GB |
| 2 (default) | 42.3 | 0.3-1.3% | 35-38% | |
| 3 | 42.1 | 0.3% | ~59% | 10.3 GB |

Each extra thread holds one more batch in flight, ~1.3 GiB of GPU memory at batch size 131072.
On this machine the GPU step is the limit, so the speed does not change; with faster decoders
(more CPUs) a single thread caps training at ~40 it/s.

### Transfer to the GPU

About 52 MB per batch reach the GPU: 48 MiB of packed feature indices (only the active features
and per-row offsets) and 1.5 MiB of floats. At 42.3 it/s that is ~2.2 GB/s, ~17% of the 13.1 GB/s
measured for pinned copies on this PCIe gen 3 x16 link. The link is not a limit.

## The GPU step

`nsys` over 10 s of steady-state training (~398 steps, ~22.6 ms of kernels per step):

| Part | Share of kernel time |
|---|---|
| FT forward (`fused_double_ft_forward`) | 30% |
| FT backward (`tiled_scatter`, `tiled_ft_prologue`) | 23% |
| AdamW (`multi_tensor_apply_kernel`) | ~21% |
| L1 / L2 / output (`_bucketed_*`) | 12% |
| Per-batch preparation from the prefetch threads | ~7% |
| Other (fills, loss, cat / clamp) | ~6% |

Some kernel runs 84% of the time. The training stream is idle 15.7% of the time, ~3.9 ms per step
in gaps of ~0.5 ms between small kernels: the step launches ~145 kernels from Python, and
between some of them the GPU waits for the host to launch the next one. Fewer loader workers
(less CPU contention) did not change the step speed.

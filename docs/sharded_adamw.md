# Experimental sharded AdamW step

`ShardedAdamWStep` connects the existing data-parallel loader to the sharded
FT, sharded L1, local layer-stack tail, weighted-loss backward, gradient
synchronization and AdamW update. It is an opt-in Python building block; the
normal `train.py`/`SimpleTrainer` and nettest checkpoint conversion do not yet
use it.

Create a process group first, set `optimizer_name` to `"adamw"`, and construct
the model directly with `NNUE(config, ..., sharding=(rank, world_size))`. Move
it to the process's device **before** constructing `ShardedAdamWStep(nnue)`;
the constructor broadcasts the replicated parameters and creates the optimizer.
Every rank must call `step(local_loader_batch)` in the same order. The returned
tensor is the global weighted loss. The caller is responsible for epoch-based
scheduler steps and the existing weight-clipping callbacks.

When the model is on CUDA, `ShardedAdamWStep` compiles the PyTorch FT weight
merge fallback, the post-L1 tail and the loss by default. Supported weights
use upstream's fused CUDA merge for fake quantization. The distributed
forward and collectives remain eager. Pass `compile_model=False` to keep
those local components eager;
CPU runs default to eager, while `compile_model=True` explicitly enables
compilation. Construct the runner after moving the model to its final device.
Compilation is lazy, and different batch shapes or quantization settings may
require additional warmup. Checkpoints can be loaded between eager and compiled
runners.

`state_dict()` is a normal shallow PyTorch snapshot: serialize or deep-copy it
before further updates. Save a separate state for every process-group rank.
`load_state_dict` requires the same rank, world size, configuration, parameter
groups, device type and clipping setting, and restores that rank's AdamW moments, scheduler,
lambda state and random generators. This format does not produce a normal
unsharded checkpoint or NNUE file; conversion and coordinated checkpoint-file
management are not implemented.

Set `jitter_lambda_batch` before constructing the runner. On multiple ranks,
the runner broadcasts an initial seed and jitter buffer. Each rank then draws
one value per step from an identical dedicated generator on its training
device, so the whole batch shares the same jitter without a per-step
collective. Per-sample jitter uses the usual local generator. The checkpoint
saves both the running jitter value and the dedicated generator state.
`--nettest-settings` enables nettest's batch jitter in the benchmark;
`--nettest-disable-batch-jitter` reproduces the older benchmark setting.

The CPU/Gloo path is for correctness testing. Eligible CUDA L1 shards use the
grouped selected-bucket kernel; unsupported shapes fall back to dense
all-bucket multiplication. Feature gathering supports the existing padded
loader tensors and an optional native-packed representation. Full
training-pipeline and checkpoint conversion behavior have not yet been
validated.

Native-packed sharded batches use the upstream loader workers to build the
packed layout. The benchmark's loader wait diagnostics include each rank's
native fetch and prefetch queue counters. Rebuild the native loader with this
checkout because the stream ABI includes the prepack flag.

Ordinary training uses upstream's tiled FT backward where supported. The
sharded path retains its existing scatter and tile-local aggregation kernels,
including for direct packed FT. The DDP benchmark uses tiled preparation in
the live prefetch thread, or prepares each replay batch inline during timing.

The benchmark's `--async-features` option experiments with one-batch
lookahead for native-packed loader batches. All ranks enqueue the next feature
collectives in the same order on the existing process group after the current
forward and before backward; a separate CUDA stream handles preparation. The
pending object retains its buffers and the consuming stream waits before
unpacking. The callback and pre-gathered batch arguments of `step` expose the
building blocks for this experiment, not a complete production pipeline.
Callers must coordinate collective ordering, batch order, stream lifetimes,
errors and shutdown across all ranks. Do not advance model weights while
preparing the current forward. See `benchmark_results.md` for measured cost
and limitations.

The runner enables the following CUDA optimizations when their guards pass.
Pass `False` to select a reference implementation and use the same settings
on all ranks. The benchmark selects these options explicitly for comparisons.

| Runner option | Benchmark flag | Behavior |
| --- | --- | --- |
| `bucketed_bias` | `--bucketed-l1-bias` | Reduce L1 bias gradients by bucket on supported NVIDIA GPUs. The CUDA path checks the dtype, shape, scratch bound, and graph capture state. |
| `uniform_loss` | `--uniform-loss` | For native-packed NVIDIA CUDA batches with uniform weights, normalize the local loss sum by the global position count from feature gathering. Weighted recipes use the usual denominator reduction. |
| `overlap_replica_gradients` | `--overlap-replica-gradients` | On NVIDIA NCCL with equal, nonempty batches and trainable FP32 replicas, enqueue replica reduction after L1 backward's all-gather and run local FT backward while it completes. Wait before gradient clipping or AdamW. |

Select `direct_packed_ft=True` to test FT consumption of gathered feature IDs
through per-row ranges. For the benchmark, use `--direct-packed-ft` with
`NNUE_BENCH_NATIVE_PACKED=1`.

Use `--repeat-live-batch --matched-batch-cache DIR --matched-batch-count 8`
to replay eight successive saved real batches in order. Choose the same
cache, starting index, batch size and world size for each variant. Cache
creation and loading are outside the timing window. Each rank holds its
replay batches on its GPU, so larger counts use more memory.

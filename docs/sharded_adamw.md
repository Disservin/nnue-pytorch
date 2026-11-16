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

`state_dict()` is a normal shallow PyTorch snapshot: serialize or deep-copy it
before further updates. Save a separate state for every process-group rank.
`load_state_dict` requires the same rank, world size, configuration, parameter
groups, device type and clipping setting, and restores that rank's AdamW moments, scheduler,
lambda state and random generators. This format does not produce a normal
unsharded checkpoint or NNUE file; conversion and coordinated checkpoint-file
management are not implemented. Batch-level lambda jitter is rejected until
its random state is synchronized across ranks.

The CPU/Gloo path is for correctness testing. Eligible CUDA L1 shards use the
grouped selected-bucket kernel; unsupported shapes fall back to dense
all-bucket multiplication. Feature gathering supports the existing padded
loader tensors and an optional native-packed representation. Full
training-pipeline and checkpoint conversion behavior have not yet been
validated.

# Multi-GPU benchmark notes

These measurements use a 131072-position global batch and a model with
`Full_Threats+PP_3Wide+K32H3^` features, FT width 1024 and 192 layer-stack
buckets. DDP uses a compiled model and loss. The sharded path is experimental
and is not yet integrated into the normal training and checkpoint pipeline.
Numbers are exploratory and do not establish training convergence or net
strength.

With a fixed, GPU-resident batch on four H100 GPUs connected by NVLink, DDP
measured 19.01 ms per step, the retained sharded path 20.50 ms, and an
experimental sharded configuration 17.94 ms. The experimental configuration
uses compiled FT merge, a bucketed bias-gradient reduction and aggregated FT
backward. On four PCIe-connected L40S GPUs, corresponding DDP and retained
sharded timings were 86.85 ms and 40.32 ms. These are different systems and
should not be treated as a controlled comparison of GPUs.

On H100, a fixed-weight diagnostic that skipped only FT gradient reductions
reduced DDP step time by 2.8–3.3 ms relative to an otherwise equivalent
all-reduce hook. The diagnostic is **not valid training**: FT gradients are
not synchronized. The measured difference includes changes in overlap and
contention and is not simply network transfer time. A single normal-DDP trace
showed 2.84–2.97 ms of NCCL gradient-reduction kernels, with small overlap
with other GPU kernels. Results depend on topology, batch and model shape.

In another four-H100 test, live loading with 64 allocated CPUs and 16 loader
workers per GPU yielded 7.06–7.08 million positions/s for DDP and 5.90–5.92
million for experimental sharding across two 1200-step windows. Both modes
had low host-side batch-acquisition wait in these runs. Disabling profiler
ranges did not close the live gap. Fixed GPU-resident measurements on the
same system were 17.99–18.73 ms for DDP and 17.68 ms for sharding, but used
per-step synchronization and a different input regime. GPU-side prefetch,
feature exchange and execution overlap need further investigation.

As a diagnostic, host timing around the sharded feature-gather call had
per-rank medians of 1.30–2.17 ms over 832 live calls and 0.79 ms over 316
fixed-input calls. The live window was asynchronous; fixed-input steps used
per-step barriers and synchronization, so these are not strictly matched.
Warmup is included, and occasional initial outliers make the mean unsuitable
for estimating steady-state latency. The results point to extra time in or
around feature distribution, but do not identify its underlying cause or
account for the whole observed throughput gap.

With eight loader workers per GPU, host batch-acquisition waits were sometimes
large and highly imbalanced between ranks. Timings vary across runs and
systems, so individual comparisons should not be taken as universal results.
The file `profile_measurements.json` contains sanitized benchmark measurement
records. It omits raw traces and environment-specific paths.

## Compilation and asynchronous feature exchange

In a follow-up with the same batch and four H100s, 64 allocated CPUs and 16
loader workers per rank, sharded training with compiled FT merge but an eager
tail and loss reached 5.39–5.46 million positions/s in two live runs. Compiling
the tail and loss increased that to 6.73–6.81 million positions/s; compiled
DDP reached 7.09 million in two runs. These are successive runs and may be
affected by data-loading variability. After fixing cross-stream batch
lifetime handling, a later 800-step comparison measured 5.28 million
positions/s with synchronous feature exchange and 5.34–5.39 million with
one-batch-ahead exchange. Earlier asynchronous measurements before that
lifetime fix should not be treated as validated training results. The sharded
FT/L1 distributed forward itself was not compiled as
one graph. An attempt to compile the entire sharded forward timed out a
four-rank NCCL correctness test with uneven local batches, so this option is
not enabled. The tests did not establish the cause of that timeout.

The asynchronous prototype submits collectives in the same order on all ranks
using one process group. It exchanges the next native-packed batch after the
current forward and before backward, waits before using the result, and keeps
the needed tensors alive across streams. It is available only in the
experimental benchmark (`--async-features`), not the normal training loop. It
increased peak per-GPU allocation by approximately 0.28 GiB. Four-rank Gloo
and NCCL tests covered uneven and empty local batches, cross-stream feature
consumption and several AdamW steps. These are smoke tests, not convergence
tests or a proof of safety for arbitrary caller schedules.

Nsight Systems captures showed recurring gaps between traced GPU activities,
including gaps before kernels launched in feature distribution. A separate
capture labels asynchronous preparation and completion. Such gaps can include
profiler effects or untraced operations, and the phase of the next launch does
not prove what caused a gap. The gap summaries are in
`nsys_gap_summary.json`, with hardware-source and filesystem metadata omitted.

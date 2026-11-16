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

The measured FT merge, tail and loss compilation are now the default for
CUDA sharded AdamW steps. The distributed forward remains eager; use the
benchmark's `--eager-sharded` switch for an uncompiled comparison.

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

## Upstream-recipe approximation

On four H100 GPUs, a later live-loader comparison used a checksum-verified
training binpack from the upstream recipe and the recipe's full remapping
binpack. The generated table retained 92 of the 192 raw layer stacks. The
benchmark used the recipe's model dimensions, feature set, position filters,
one-cycle AdamW parameters, sample jitter, per-step dense weight clipping,
shuffle-buffer size and one PyTorch CPU thread per rank. It ran after the
fake-quantization warmup, with a global batch of 131072. **Batch-level lambda
jitter was disabled in both modes**, because the sharded runner does not yet
support it; this is therefore not an exact training run. It used one training
file rather than the full multi-file mixture.

The host allocation had 64 CPUs. Upstream's 32 loader workers per rank would
produce 128 loader workers, so the comparison instead used 8, 12 or 16 per
rank. Each number below is a 600-step live window; positions vary between
runs.

| Trainer | Loader workers/rank | Positions/s | Peak allocated/GPU |
| --- | ---: | ---: | ---: |
| Compiled DDP | 12 | 4.80–4.81M | 4.08 GiB |
| Compiled sharded | 8 | 4.04M | 1.73 GiB |
| Compiled sharded | 12 | 4.66–4.73M | 1.73 GiB |
| Compiled sharded, async features | 12 | 4.72M | 1.99 GiB |
| Compiled sharded | 16 | 4.68M | 1.73 GiB |
| Compiled DDP | 16 | 5.89M | 4.08 GiB |

The earlier benchmarks used weaker default filtering, a smaller shuffle
buffer, four PyTorch threads per rank, and did not include the same per-step
weight clipping and scheduler. Thus the earlier higher throughput is not
directly comparable. Controlled ablations are needed to determine the cause
of the difference. The upstream remapping table used here is saved in
`nettest_stack_remap.txt`; the corresponding benchmark requires explicit
acknowledgment that batch jitter is disabled.

On a separate four-H200 NVL system with 64 allocated CPUs and 128 visible
hardware threads, a same-node follow-up used 16 loader workers per rank. DDP
reached 8.49–8.71M positions/s and compiled sharding 7.60–7.94M; one
asynchronous sharded run reached 8.02M. One-variable sharded controls reached
8.22–8.51M when per-step weight clipping was disabled, 7.78–7.85M with the
earlier benchmark's filters, 7.82–7.93M with four PyTorch CPU threads per
rank, and 7.89–8.13M with the smaller shuffle buffer. These ranges are from
two 600-step runs per configuration, except the baseline has three and the
asynchronous variant one. The filter, thread-count and shuffle
results overlap the baseline range; disabling weight clipping changed
training behavior and is only a diagnostic.

With the same H200 workload and settings, the `--eager-sharded` opt-out
reached 6.41–6.88M positions/s in two runs. Thus the default compilation
improved this comparison, although the windows still show variability.

With 12 loader workers per rank on the H200 system, one DDP run reached 6.91M
positions/s and a rank spent 5.60 seconds acquiring batches during an
11.38-second measurement window. Its 16-worker run spent substantially less
host time acquiring batches and reached 8.49M, indicating that loader
imbalance can significantly limit throughput. Host acquisition time does not
measure all GPU-side waits. The H200 and H100 systems differ, so these results
do not isolate the cause of the H100 slowdown.

Restricting the H200 job to 32 physical cores / 64 hardware threads on the
same node yielded 8.38M positions/s for DDP and 6.99M for sharding with 16
loader workers per rank (one run each). At 12 workers, the respective figures
were 7.44M and 7.47M. CPU affinity and concurrent activity can affect these
results; the restricted-core run is not a reproduction of the H100 host.

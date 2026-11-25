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
jitter was disabled in both modes** because the sharded runner did not
support it at the time. This comparison used one training file from the
full multi-file mixture.

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
`nettest_stack_remap.txt`. Pass `--nettest-disable-batch-jitter` to reproduce
this comparison's jitter setting with the current benchmark.

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

### Four-GPU detailed profiling follow-up

On four H200 NVL GPUs, the live-loader benchmark used the settings above,
including batch 131072, 92 stored stacks and 16 workers per rank, with
32 warmup and 600 measured steps per run. It uses one real binpack and the
real stack remap rather than the complete training data mixture. Batch lambda
jitter was disabled on both paths. Results are individual unprofiled windows:

| Configuration | M positions/s | ms/global batch |
| --- | ---: | ---: |
| Compiled DDP | 8.30–8.51 (2) | 15.39–15.80 |
| Compiled sharding, default clipping | 6.85, 7.68–7.80, 7.76 (4) | 16.81–19.14 |
| Sharding, experimental foreach clipping | 8.40–8.50 (2) | 15.43–15.61 |
| Sharding, gradient clipping disabled | 9.06–9.07 (2) | 14.45–14.47 |
| DDP, gradient clipping disabled | 9.33 (1) | 14.05 |
| Sharding, weight clipping disabled | 7.67–8.18 (2) | 16.02–17.09 |
| Sharding, experimental bucketed L1 bias gradient | 8.02–8.29 (2) | 15.81–16.35 |
| Sharding, asynchronous feature distribution | 7.37–7.82 (2) | 16.77–17.79 |
| Sharding, direct FT backward | 7.09–7.37 (2) | 17.79–18.49 |

The first sharded baseline was an unusually slow window; subsequent baselines
bracketed the candidate runs. The clipping-disabled controls change training
and are diagnostic only. Individual changes and differences between these
windows are not additive. The FT-backward comparison is specific to this GPU,
shape and the current automatically selected aggregated kernel. See
`detailed_profile_results.json` for individual windows.

Nsight Systems captured 100 live steps on each rank for compiled DDP, default
sharding and foreach-clipping sharding. These traces are intrusive: the default
sharded trace ran at 5.11M positions/s, so the unprofiled numbers above are the
throughput evidence. The median host span for sharded gradient clipping was
1.64–1.99 ms across ranks, with 193 attributed kernel launches per step;
the experimental implementation reduced this to 0.61–0.74 ms and 27 launches.
DDP's clipping span was 0.54–0.66 ms and 24 launches. These are nested host
spans, not isolated GPU durations. They cannot be summed into a step breakdown.
The experiment batches local gradient norm and scale operations, retaining
the scalar cross-rank reduction and counting replicated parameters once.
Four-rank tests compared norms and gradients for clipping, no clipping, zero
and absent gradients on CPU and GPU, and compared multi-step AdamW loss and
state on GPU. The batched implementation is now the default in
`clip_sharded_grad_norm_`; the results above measured its experimental
predecessor, before the production version retained additional API checks and
support for ranks with no gradients.

A subsequent four-H200 run validated the production implementation with
four-rank NCCL norm tests, a two-GPU AdamW comparison to the unsharded model,
and a compiled four-rank multi-step test. An isolated clipping benchmark with
the real 92-stack parameter shapes and synthetic, reused gradients measured
the original loop at 0.96–1.01 ms and the production loop at 0.26–0.27 ms
per call (two 300-call windows each). This is not an end-to-end speedup; the
benchmark is reproducible with `scripts/benchmark_sharded_clip.py`.

On the same system, two 1800-step live-loader runs with the original clip
reached 8.05–8.11M positions/s, while two with the new clip reached
7.81–7.88M. The slowest rank spent 0.36–0.52 seconds acquiring batches in the
original runs versus 1.89–2.87 seconds in the new runs. Acquisition and GPU
work can overlap, and these timings do not isolate cause; they do show that
the end-to-end comparison is affected by loader imbalance. Shorter 600-step
windows varied more widely and do not establish a stable throughput gain on
this system. Both paths retained gradient and weight clipping.

The default sharded trace also showed a 5.08–5.84 ms median host span in
replicated-gradient synchronization and a 1.52–2.40 ms median feature-gather
span. These include asynchronous work and waits for earlier GPU operations;
they do not establish the communication cost in isolation. In particular,
replica synchronization checks gradient presence using a host read, after
backward has been enqueued. DDP overlaps its gradient reductions with
backpropagation. Sharding also needs L1 forward reduce-scatter and backward
all-gather, so the two communication schedules differ. Investigating those
dependencies and host synchronizations is a promising next step.

The traced aggregated FT backward kernel averaged 3.44–3.46 ms per step
across sharded ranks and 3.53–3.65 ms for DDP; traced FT feature packing was
0.59 ms per sharded rank versus 0.15 ms per DDP rank. Each sharded rank sees
the full batch, while each DDP rank sees a quarter; FT output width is divided
across sharded ranks. These are device-kernel durations under instrumentation,
may overlap other work, and are not additive estimates of exposed step time.
They suggest reducing repeated full-batch feature preparation is worth
investigating in addition to removing host synchronization.

In the same instrumented default-sharding trace, launch attribution on rank
0 counted 54 kernels per step in the model weight-clipping callback, 37 in
sharded L1 forward, and 30 in the fused AdamW step. DDP also launched 54 in
weight clipping. These counts include nested calls and do not establish that
launch overhead is exposed; the weight-clipping ablation above also changes
training behavior and its measured range overlaps the baseline. Weight
clipping performs grouped/virtual-weight expansion before clamping; a fused
implementation preserving those bounds is a candidate for a controlled test.
The L1-forward count includes elementwise and gather operations for its
factorized weights, as well as the grouped matmul and collective. Any change
should be compared against the existing grouped L1 path with real stack maps.

The trace's median host loader-acquisition span was 0.09–0.11 ms per rank in
the sharded run; the 600-step baseline measured 0.11–0.17 seconds of host
acquisition per rank. This does not include all GPU-side input waits. Raw
traces and logs contain machine-specific metadata and should remain private.

### Dense clipping and L1 launch reduction

The frequent dense-layer weight clipping is compiled for CUDA FP32 when
training with Inductor. Feature-transformer clipping remains a separate eager
operation. Sharded training also compiles the L1 factor merge and, for equal
nonempty NCCL batches, directly reduce-scatters the partial output and uses
an all-gather into a contiguous tensor for backward. Uneven, empty, Gloo and
single-rank batches retain the padded implementation. Compiled code falls back
to eager for CPU, autocast or non-FP32 parameters; the sharded runner's
`compile_model=False` disables both compiled helpers. Compiled callables are
unbound and cached outside the model so model copies and serialization do not
capture the original model instance.

On four H200 GPUs, compiled QAT, batch 131072, 92 stored stacks, and the real
binpack and remap were used for the comparisons below. Each measurement has
32 warmup and 600 measured steps; both paths include gradient and weight
clipping. Lambda batch jitter remains disabled, and the binpack is one source
from the training mixture. The original sharded control explicitly uses the
old padding collective and eager dense clipping/L1 merge while retaining the
previously compiled FT, tail, loss and batched gradient clipping. The DDP
control differs only in dense clipping. Two runs were performed for each.

| Case | Input | Before (M pos/s; ms/batch) | After (M pos/s; ms/batch) |
| --- | --- | --- | --- |
| Sharded | live, 16 loader workers/rank | 8.46–8.52; 15.39–15.49 | 9.26–9.35; 14.02–14.16 |
| Sharded | repeated identical real batch | 10.70–10.79; 12.15–12.25 | 11.32–11.35; 11.55–11.58 |
| DDP | live, 16 loader workers/rank | 8.10–8.83; 14.85–16.18 | 8.39–8.45; 15.51–15.62 |
| DDP | repeated identical real batch | 9.68–9.91; 13.23–13.54 | 10.06–10.07; 13.02–13.03 |

The repeated-batch diagnostic saves the first rank-local batches and reloads
the same data within each mode; DDP and sharding use separately captured
batches. It excludes ongoing data acquisition and is not a training-throughput
result. In the live sharded windows, the slowest rank's
host acquisition time was 0.11–0.35 seconds per run. The slowest DDP rank
spent 2.96–3.90 seconds acquiring data in each roughly nine-second window;
the measured spans can overlap GPU work. The live DDP ranges overlap and do
not establish either a DDP improvement or a fair DDP-versus-sharding speed
advantage. Individual run data are in `kernel_launch_results.json`.

Nsight captures on the repeated batch attributed 54 kernels per step to dense
weight clipping before versus 3 after, and 37 kernels in sharded L1 forward
before versus 10 after, on rank 0. These are nested CPU-launch attributions;
tracing perturbs execution and the counts do not measure exposed time. Separate
repeated-batch experiments with all three changes together were 11.53–11.57 ms
versus 12.02–12.27 ms in their baseline windows. Isolating changes showed
smaller gains and interactions, so the differences should not be added.

GPU correctness checks covered QAT on/off, rounding boundaries, cancellation,
zero/one/multiple nonzero factor maps, remapped stacks, clipping boundary and
non-finite values, copied/serialized models and CPU/autocast fallback. The
real 92-stack, three-factor model matched eager clipping and L1 merge exactly.
Distributed tests compared original and optimized collectives and optimizer
steps, including equal, uneven, empty, noncontiguous and subgroup cases. CUDA
graph capture/replay of these new paths has not been verified.

### Matched-batch L1 bias reduction versus DDP

On four NVLink-connected H200 GPUs, 64 allocated CPU cores (128 visible
hardware threads), and 16 loader workers per rank, three paths were compared:
compiled DDP, the current compiled sharded runner, and the same sharded runner
with the experimental bucketed L1 bias-gradient reduction. Each used global
batch 131072, 92 stored stacks, post-warmup fake quantization, AdamW, gradient
and weight clipping, one PyTorch CPU thread per rank, the nettest filters,
and a real training binpack. Lambda batch jitter was disabled in all paths;
the binpack is one part of the real training mixture.

The fixed-input diagnostic saved two distinct real global batches. For each,
DDP and sharding loaded the same dense rank-local positions and targets;
sharding converted those to native packed form before timing. Shards were
initialized by slicing the same seed-initialized full model as DDP. Thus
inputs and starting parameters match, though optimizer trajectories can differ
after updates. Each run included 48 warmup and 600 measured steps, repeatedly
using one saved batch; ongoing data loading was excluded. There were three
runs per path on the first batch and two per path on the second.

| Path | Matched-batch positions/s | ms/global batch |
| --- | ---: | ---: |
| Compiled DDP | 9.26–9.41M | 13.92–14.15 |
| Current compiled sharding | 10.52–10.60M | 12.37–12.45 |
| Sharding with bucketed bias gradient | 11.41–11.57M | 11.33–11.48 |

These results establish a repeated-input compute comparison for the tested
positions and settings, not full live-training throughput. Peak allocated GPU
memory in these runs was 3.87–4.07 GiB for DDP and 1.895 GiB for both sharded
paths; this measures peak allocated, not reserved memory.

For live loading, each path ran three 1000-step windows with 48 warmup steps,
the same initial model seed, and independently loaded positions:

| Path | Live positions/s | ms/global batch | Slowest-rank host acquisition per run |
| --- | ---: | ---: | ---: |
| Compiled DDP | 8.02–8.65M | 15.15–16.35 | 3.96–7.73 s |
| Current compiled sharding | 8.14–8.72M | 15.03–16.10 | 0.87–0.95 s |
| Sharding with bucketed bias gradient | 7.54–8.94M | 14.67–17.39 | 1.30–2.44 s |

Host acquisition may overlap GPU work, so its duration cannot simply be
subtracted from the window. The DDP runs show substantial loader imbalance,
and the live ranges overlap. The live runs do not establish a consistent
end-to-end winner or an improvement from the bucketed reduction.

The bucketed experiment passed CUDA forward and gradient comparisons for
92 and 192 stacks and batches up to 32768, concentrated-bucket and
noncontiguous-gradient cases, and a four-rank compiled multi-step comparison
of gradients, losses, optimizer state and parameters. It remains an opt-in
benchmark override. Before promoting it, guard its supported device, dtype,
index and stack-count contract and bound its `count * local_batch * 4` byte
routing scratch allocation. Individual sanitized results are stored in
`bias_comparison_results.json`; raw logs stay private.

### Shared L1 factor ablation

This diagnostic removes the shared `factorized_linear` weight and bias from
L1, while retaining all L1 group factors and the FT factorization. It folds
the shared factor into each base stack first, preserving the initial effective
weights, then freezes and zeros the two retained placeholder parameters. It
omits them from optimizer groups, sharded gradient synchronization and norm
clipping, skips their forward merge, and removes their virtual weight from
dense weight clipping. This is a changed training parameterization, not an
equivalent long-term training run; the factor removal is benchmark-only via
`--ablate-shared-l1-factor`.

On four NVLink-connected H200 GPUs with 64 allocated CPU cores and 16 loader
workers per GPU, the test used global batch 131072, 92 stored stacks, QAT,
the same upstream settings and limitations as the preceding comparison, one
real binpack, and disabled lambda batch jitter. Two distinct saved global
batches and matching seed-initialized full-model weights were used; each
configuration ran twice per batch with 48 warmup and 600 measured steps.
The fixed-input timing excludes ongoing data loading:

| Path | Shared factor | Positions/s | ms/global batch, mean (range) |
| --- | --- | ---: | ---: |
| Compiled DDP | On | 9.68–10.05M | 13.286 (13.045–13.535) |
| Compiled DDP | Removed | 10.08–10.09M | 13.002 (12.997–13.009) |
| Compiled sharding | On | 11.39–11.46M | 11.467 (11.436–11.507) |
| Compiled sharding | Removed | 11.31–11.49M | 11.494 (11.411–11.588) |
| Sharding + experimental bucketed bias | On | 12.30–12.45M | 10.604 (10.532–10.659) |
| Sharding + experimental bucketed bias | Removed | 12.26–12.48M | 10.594 (10.500–10.694) |

The sharded timings overlap; neither sharded variant shows a measurable saving
at this resolution. DDP saved about 0.28 ms per batch on the means, or around
2%, with noticeable baseline variation and only four runs per configuration.
On the live loader, two 1000-step windows per configuration also overlapped:
DDP 7.88–8.95M with the factor versus 8.28–9.09M without; sharding 8.19–8.95M
versus 8.40–9.24M; and bucketed-bias sharding 8.28–8.87M versus 8.19–9.40M.
Loader acquisition varied, so these live ranges do not establish a small gain.

The shared factor is only 32,768 weights plus 32 biases in the full model,
or 8,192 weight values plus 32 replicated biases per rank for four-way
sharding. The retained placeholder weights mean the ablation's measured
allocated memory does not represent fully deleting those tensors. In a
100-step rank-0 Nsight capture, L1 forward and dense clipping had the same
attributed launch counts in the bucketed-bias sharded path with and without
the factor (10 and 3 per step), while AdamW fell from 30 to 26 launches.
These are instrumented, nested CPU-launch attributions, not additive timings.

Tests compared initial outputs, QAT, group-factor gradients and optimizer
membership with nonzero folded factors on both full and sharded models; the
full model's reconstructed per-stack L1 weights were also compared. A
separate four-rank NCCL test compared first-step losses,
gradients, remaining parameter updates and AdamW states, including an empty
local batch. No net-strength or convergence comparison was performed.
Individual sanitized windows are in `l1_factor_ablation_results.json`.

### Guarded bucketed L1 bias reduction

These runs selected the bucketed bias-gradient path explicitly through
`ShardedAdamWStep(..., bucketed_bias=True)` or the benchmark's
`--bucketed-l1-bias`. It keeps the eager PyTorch
forward gather and bounds behavior, accepts valid negative indices, masks
invalid router accesses before narrowing int64 IDs, and does not read index
values back to the host. It only selects the fast CUDA path for supported
same-device FP32 shapes, int64 indices, 1–256 stacks, width 1–128 and a
conservative 64 MiB estimated backward-scratch cap. The estimate includes
router rows/counts, reduction partials and result, plus possible contiguous
copies of indices and incoming gradients; it excludes the common eager forward
outputs and allocator overhead. For 92 stacks, 32768 local positions and width
32, the conservative estimate is approximately 15.85 MiB per GPU.
Unsupported devices, dependencies, shapes,
dtypes, autocast, deterministic mode, empty inputs and CUDA graph capture use
the existing eager path. Higher-order gradients use differentiable PyTorch
reduction rather than the raw kernel. The caller chooses the option again
when constructing a runner for checkpoint resume.

On four H200 GPUs with 64 allocated CPU cores and 16 loader workers per rank,
the final graph-safe version was compared with current sharding. Both used
global batch 131072, 92 stored stacks, post-warmup QAT, AdamW, clipping,
nettest filters, one real training binpack and identical initial model weights.
Lambda batch jitter was disabled. Two different saved global batches were
used for one 1000-step timing per path and batch, after 64 warmup steps; the
same batch was reused during each window, excluding ongoing loading:

| Path | Matched-batch positions/s | ms/global batch |
| --- | ---: | ---: |
| Current sharding | 11.44–11.51M | 11.393–11.460 |
| Guarded bucketed bias | 12.35–12.39M | 10.576–10.613 |

The saved-batch path improved throughput by roughly 8% and reduced step time
by about 0.8 ms. A 100-step Nsight capture of the final version contained
exactly 100 router and 100 bias-reduction kernels on each rank. This verifies
execution rather than relying only on the benchmark's metadata eligibility
probe. Tracing itself is intrusive and is not used for throughput numbers.

In two final-version 3000-step live windows per path, with 64 warmup steps and
independently loaded positions, current sharding reached 10.23–10.25M pos/s
(12.787–12.815 ms/batch), while bucketed bias reached 10.10–10.18M pos/s
(12.882–12.977 ms/batch). The slowest rank spent 0.62–1.00 s per baseline run
and 1.73–2.09 s per bucketed run acquiring batches; that host time can overlap
GPU work. Four additional 3000-step windows per path on an earlier guard
iteration varied more widely: 8.81–10.25M baseline and 9.92–10.64M bucketed.
The code versions and individual windows are identified in
`bucketed_bias_hardened_results.json`. Live throughput varied by run.

The final GPU suite passed 37 tests, including the original bias-gradient
tests, concentrated buckets, noncontiguous indices/gradients, negative and
invalid int64 IDs, extreme values, memory-cap boundaries, fallback cases,
higher-order derivatives, CUDA graph forward capture/replay, nondefault
streams, cross-device execution and four-rank compiled AdamW parity. A
separate four-rank test compared two QAT/AdamW steps on the real 92-stack
model and saved batch, including gradients, parameters and optimizer states.
These are correctness tests, not convergence or net-strength measurements.

### PCIe-only host: sharded training from train.py

This system has two A100-PCIE-40GB GPUs on PCIe 3.0 x16 with no NVLink. The
visible link is Gen3 x16, `nvidia-smi topo` reports a host-bridge (NODE)
connection, and a two-rank fp32 NCCL all-reduce sustains about 15.5 GB/s
aggregate: 565 MiB in 76.3 ms, 283 MiB in 38.3 ms, 141 MiB in 19.2 ms.

The threats recipe's model has 542 MiB of feature-transformer parameters
(59808x1024, 4560x1024, 73728x1024 and 768x1024 virtual). DDP reduces all of
them at the end of backward, after the last FT gradient is produced, so
almost none of that time overlaps compute. The recipe's validation and
optimizer settings otherwise match `scripts/nettest_benchmark_settings.py`.

Run command: `torchrun --standalone --nproc-per-node=N ddp_launcher.py
train.py <recipe options>` with the 60-file training mixture, global batch
131072, `--epoch-size 52428800` (400 steps), 32 loader workers per rank,
`--threads 1`, and `--num-workers`/affinity set by the launcher. The sharded
run adds `--sharded`. Numbers are the slope between 80% and 100% of the run,
after compilation; three-second windows make them approximate.

| Configuration | it/s (131072-pos batches) | ms/batch |
| --- | ---: | ---: |
| One A100, DDP | 26.7 | 37 |
| Two A100, DDP | 10.0 | 100 |
| Two A100, `--sharded` | 26.7 | 37 |

Single-GPU numbers are loader-bursty (the queue ran dry on 78% of fetches at
32 workers) but the GPU was the limiting side in the measured window. Both
multi-GPU runs had a full device queue, so DDP is GPU/communication bound.
The sharded run uses the guarded bucketed L1 bias reduction and dense replica
gradient overlap by default. Its communication per step is dominated by the
feature exchange: each rank's packed active features (~50 MiB for this
feature set and batch at ~190 active features/position) are all-gathered so
both ranks can run their FT column shard on every position. It also has one
16.8 MiB L1 reduce-scatter, one 16.8 MiB gradient all-gather and the
replicated tail reduction.

The earlier benchmark-script measurement on the same host and data, with
bucketed bias disabled and replica overlap explicitly off, was 22.2 it/s
(2.91M positions/s); enabling replica-overlap reached 25.4 it/s. Running the
same training loop from `train.py` with both defaults reaches 26.7 it/s.

Checkpointing: `--sharded` writes a full-model checkpoint (588 MiB for this
recipe) on rank 0 plus one 882 MiB `<path>.rankN` file per rank with shard
weights, AdamW state, schedulers and RNG. A 40-step smoke run with validation
and a second epoch resumed from those files at the same throughput.

This comparison is a throughput measurement. The existing two-rank NCCL test
compares sharded and full-model AdamW gradients, losses, parameters and
optimizer state; no convergence or net-strength comparison was performed.

### Can two GPUs beat one on this host?

At the recipe's global batch of 131072 they do not: sharded training matches
one GPU (26.7 it/s) rather than exceeding it. A two-rank Nsight Systems
capture attributes about 6 ms/step to the FT forward and tiled/aggregated
backward at half width, 2-3 ms to the L1 reduce-scatter, and about 8 ms to
the three all-gathers, plus the same optimizer, clipping and host overheads
as one GPU. The compute saved by halving FT/L1 width is therefore spent on
the feature and activation exchange.

Measured attempts to close that gap, each with the same recipe and data:

| Change | Result |
| --- | --- |
| Tiled FT backward instead of the aggregated kernel, prep built at gather time | FT backward kernel 11.6 -> 4.2 ms, but the tile preparation (sort + one host sync) cancels the gain; 26.7 it/s |
| Same, with a sync-free preparation sized from the gather's host entry count | global sort costs what the tile sort's sync cost; 26.7 it/s |
| Uniform loss normalization when `w1 == 0` (drops the denominator all-reduce) | 26.7 it/s; the reduction was already overlapped |
| Host-size exchange over Gloo with no NCCL `wait()` | invalid: NCCL output readiness is not stream-ordered, so the unpack raced and faulted |
| Global batch 262144 (per-GPU batch 131072) | two GPUs 13.3 it/s of 262144 = 3.49M positions/s; one GPU 10.0 it/s of 262144 = 2.62M positions/s; one GPU at 131072 = 3.5M positions/s |

So two GPUs are 1.33x one GPU when both are given a 262144 batch, but they
still do not exceed one GPU running the recipe's 131072 batch in
positions/s. The remaining lever is the size of the feature exchange
(16-bit ID packing, including a separate escape for the 17-bit K32H3 rows,
would save roughly 5-10%); removing the L1 gradient all-gather would change
each shard's gradients to local positions only, which the parity tests in
this branch deliberately avoid.

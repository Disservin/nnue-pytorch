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

With eight loader workers per GPU, host batch-acquisition waits were sometimes
large and highly imbalanced between ranks. Timings vary across runs and
systems, so individual comparisons should not be taken as universal results.
The file `profile_measurements.json` contains sanitized benchmark measurement
records. It omits raw traces and environment-specific paths.

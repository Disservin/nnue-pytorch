"""Time eager DDP versus FT/L1-sharded AdamW with synthetic sparse features.

Run separately for each mode with torchrun --standalone --nproc-per-node=2.
Times exclude data loading and initialization and include forward, backward,
gradient synchronization, clipping, and the optimizer update.
"""

import argparse
import functools
import json
import os
import statistics
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data_loader import (
    DataloaderSkipConfig,
    FixedNumBatchesDataset,
    SparseBatchDataset,
    SparseBatchProvider,
)
from data_loader.config import DataloaderDDPConfig, DataloaderIOConfig
from model.config import ModelConfig, NNUEConfig
from model.nnue import NNUE
from model.optimizers import OptimizerConfig
from trainer.sharded_adamw import ShardedAdamWStep


def make_batch(size, device, rank):
    width = 260
    row = torch.arange(size, device=device, dtype=torch.int64).reshape(-1, 1)
    col16 = torch.arange(16, device=device, dtype=torch.int64).reshape(1, -1)
    col32 = torch.arange(32, device=device, dtype=torch.int64).reshape(1, -1)
    features = []
    for perspective in range(2):
        indices = torch.full((size, width), -1, dtype=torch.int32, device=device)
        shift = rank * 173 + perspective * 503
        indices[:, :16] = (row * 13 + col16 * 97 + shift).remainder(59808).int()
        indices[:, 16:32] = (
            59808 + (row * 3 + col16 * 17 + shift).remainder(4560)
        ).int()
        indices[:, 32:64] = (
            64368 + (row * 7 + col32 * 101 + shift).remainder(73728)
        ).int()
        features.append(indices)
    us = row.remainder(2).float()
    outcome = row.remainder(3).float()
    score = (row.remainder(101).float() - 50) * 20
    stacks = row.flatten().remainder(192)
    return us, 1 - us, *features, outcome, score, stacks


def pack_saved_batch(batch):
    """Create native-layout input from a saved dense batch, outside timing."""
    us, them, white, black, outcome, score, stacks = batch
    n, width = white.shape
    wm, bm = white >= 0, black >= 0
    wi, bi = white[wm], black[bm]
    zero = torch.zeros(1, dtype=torch.int32, device=white.device)
    wo = torch.cat((zero, wm.sum(1, dtype=torch.int32).cumsum(0, dtype=torch.int32)))
    bo = torch.cat((zero, bm.sum(1, dtype=torch.int32).cumsum(0, dtype=torch.int32)))
    native = {"format": "native_offsets_v1", "packed": torch.cat((wo, bo, wi, bi, stacks.int())),
              "size": n, "width": width, "nw": wi.numel(), "nb": bi.numel()}
    return us, them, native, None, outcome, score, stacks


def load_matched_batches(loader, cache, first, count, rank, global_batch, device):
    os.makedirs(cache, exist_ok=True)
    paths = {
        i: os.path.join(cache, f"batch-{i}-{rank}-{global_batch}.pt")
        for i in range(first, first + count)
    }
    missing = {i for i, path in paths.items() if not os.path.isfile(path)}
    if missing:
        for i in range(max(missing) + 1):
            batch = loader[i]
            if i in missing:
                # Save CPU tensors so the cache can be replayed on any device.
                torch.save(tuple(t.cpu() for t in batch), paths[i])
    return tuple(
        torch.load(path, map_location=device, weights_only=True) for path in paths.values()
    )


def match_sharded_initial_parameters(nnue, config, *, max_epoch, num_batches_per_epoch, seed):
    """Initialize shards from the same full-model seed as the DDP control."""
    torch.manual_seed(seed)
    full = NNUE(config, max_epoch=max_epoch, num_batches_per_epoch=num_batches_per_epoch)
    nnue.model.input.load_unsharded(full.model.input)
    nnue.model.layer_stacks.l1.load_unsharded(full.model.layer_stacks.l1)
    nnue.model.layer_stacks.l2.load_state_dict(full.model.layer_stacks.l2.state_dict())
    nnue.model.layer_stacks.output.load_state_dict(full.model.layer_stacks.output.state_dict())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("ddp", "sharded"))
    parser.add_argument("--global-batch", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--binpack", nargs="+", help="Use native-loader binpack files rather than synthetic features"
    )
    parser.add_argument("--loader-workers", type=int,
                        help="Native loader workers per rank (default: 8; nettest settings: 32)")
    parser.add_argument("--nettest-settings", action="store_true",
                        help="Use the upstream nettest model, optimizer and filter settings")
    parser.add_argument("--nettest-remap-data", help="Nettest's stack-remapping binpack")
    parser.add_argument("--nettest-remap-cache", help="Read or create a cached stack-remapping table")
    parser.add_argument("--nettest-disable-batch-jitter", action="store_true",
                        help="Disable batch lambda jitter for comparison with older benchmarks")
    parser.add_argument("--nettest-ablate-filters", action="store_true",
                        help="Use the earlier benchmark's filtering for a controlled comparison")
    parser.add_argument("--nettest-torch-threads", type=int,
                        help="Override nettest's one PyTorch CPU thread per rank")
    parser.add_argument("--nettest-shuffle-entries", type=int,
                        help="Override the upstream loader shuffle-buffer size")
    parser.add_argument("--nettest-ablate-weight-clipping", action="store_true",
                        help="Skip per-step weight clipping for a controlled comparison")
    parser.add_argument("--ablate-gradient-clipping", action="store_true",
                        help="Skip gradient-norm clipping; diagnostic only, not equivalent training")
    parser.add_argument("--ablate-shared-l1-factor", action="store_true",
                        help="Benchmark removing the shared L1 weight/bias factor (changes training)")
    parser.add_argument("--live-loader", action="store_true",
                        help="Include asynchronous binpack loading in each timed step")
    parser.add_argument("--repeat-live-batch", action="store_true",
                        help="Diagnostic: replay cached inputs throughout the timing window")
    parser.add_argument("--repeat-batch-cache",
                        help="Save or reload repeated rank batches; use a fresh directory per input and configuration")
    parser.add_argument("--matched-batch-cache",
                        help="Cache the same dense rank batches for DDP and packed sharding comparisons")
    parser.add_argument("--matched-batch-index", type=int, default=0,
                        help="Select a later loader batch when creating a matched cache")
    parser.add_argument("--matched-batch-count", type=int, default=1,
                        help="Cycle through successive matched batches in one timed window")
    parser.add_argument("--matched-initial-parameters", action="store_true",
                        help="Initialize shards from the same full-model seed as DDP")
    parser.add_argument("--throughput-window", action="store_true",
                        help="Time a sustained live-loader window without per-step barriers")
    parser.add_argument("--measure-loader-wait", action="store_true",
                        help="Record host-side batch acquisition time during throughput windows")
    parser.add_argument("--measure-input-device-wait", action="store_true",
                        help="Measure the GPU stream interval surrounding batch acquisition")
    parser.add_argument("--async-features", action="store_true",
                        help="Experimentally overlap next-batch feature exchange with backward")
    parser.add_argument("--bucketed-l1-bias", action="store_true",
                        help="Use the optional guarded bucketed L1 bias-gradient path")
    parser.add_argument("--require-bucketed-l1-fastpath", action="store_true",
                        help="Fail rather than measure an unsupported bias fast-path configuration")
    parser.add_argument("--uniform-loss", action="store_true",
                        help="Experimentally skip the denominator reduction for uniform weights")
    parser.add_argument("--overlap-replica-gradients", action="store_true",
                        help="Experimentally overlap replica gradient reduction with FT backward")
    parser.add_argument("--direct-packed-ft", action="store_true",
                        help="Experimentally consume native packed feature ranges in FT")
    parser.add_argument("--nsys-capture", action="store_true",
                        help="Bracket the measured throughput window with CUDA profiler APIs")
    parser.add_argument(
        "--save-batch-dir", help="Save decoded rank batches for an identical comparison"
    )
    parser.add_argument("--load-batch-dir", help="Reuse previously saved rank batches")
    parser.add_argument(
        "--compile-ddp",
        action="store_true",
        help="Compile the DDP model and loss as train.py does",
    )
    parser.add_argument("--eager-sharded", action="store_true",
                        help="Disable the default compilation of sharded local components")
    parser.add_argument(
        "--profile", action="store_true", help="Profile one additional step"
    )
    parser.add_argument("--profile-all-ranks", action="store_true",
                        help="Save a profiler trace for every rank")
    parser.add_argument("--ddp-gradient-mode", default="normal",
                        choices=("normal", "hook", "skip-ft", "skip-all"),
                        help="Diagnostic DDP communication mode; skip modes are not valid training")
    parser.add_argument("--ddp-freeze-weights", action="store_true",
                        help="Set DDP optimizer learning rates to zero for fixed-weight diagnostics")
    args = parser.parse_args()
    if args.nettest_settings and not args.nettest_remap_data and not (
        args.nettest_remap_cache and os.path.isfile(args.nettest_remap_cache)
    ):
        raise ValueError("nettest settings require --nettest-remap-data or an existing remap cache")
    if args.nettest_settings and (not args.live_loader or args.global_batch not in (131072, 65536)):
        raise ValueError("nettest settings require a live batch of 131072 or 65536")
    if not args.nettest_settings and (args.nettest_disable_batch_jitter or args.nettest_ablate_filters
                                      or args.nettest_torch_threads is not None
                                      or args.nettest_shuffle_entries is not None or args.nettest_ablate_weight_clipping):
        raise ValueError("nettest ablations require --nettest-settings")
    if args.nettest_torch_threads is not None and args.nettest_torch_threads < 1:
        raise ValueError("--nettest-torch-threads must be positive")
    if args.nettest_shuffle_entries is not None and args.nettest_shuffle_entries < 1:
        raise ValueError("--nettest-shuffle-entries must be positive")
    if args.loader_workers is None:
        args.loader_workers = 32 if args.nettest_settings else 8
    if args.compile_ddp and args.mode != "ddp":
        raise ValueError("--compile-ddp applies only to the DDP baseline")
    if args.eager_sharded and args.mode != "sharded":
        raise ValueError("--eager-sharded applies only to sharded mode")
    if args.bucketed_l1_bias and args.mode != "sharded":
        raise ValueError("--bucketed-l1-bias applies only to sharded mode")
    if args.require_bucketed_l1_fastpath and not args.bucketed_l1_bias:
        raise ValueError("--require-bucketed-l1-fastpath requires --bucketed-l1-bias")
    profile_compile = any(os.environ.get(key) == "1" for key in (
        "NNUE_PROFILE_COMPILE_FT_MERGE", "NNUE_PROFILE_COMPILE_TAIL", "NNUE_PROFILE_COMPILE_LOSS"
    ))
    if args.eager_sharded and profile_compile:
        raise ValueError("--eager-sharded cannot be combined with profiling compilation overrides")
    if args.ddp_gradient_mode != "normal" and args.mode != "ddp":
        raise ValueError("DDP communication diagnostics require ddp mode")
    if args.ddp_freeze_weights and (args.mode != "ddp" or args.live_loader):
        raise ValueError("fixed-weight diagnostics require DDP and fixed batches")
    if args.ddp_gradient_mode != "normal" and args.live_loader:
        raise ValueError("DDP communication diagnostics require fixed benchmark batches")
    if args.profile_all_ranks and not args.profile:
        raise ValueError("--profile-all-ranks requires --profile")
    if args.load_batch_dir and args.binpack:
        raise ValueError("choose either --binpack or --load-batch-dir")
    if args.save_batch_dir and not args.binpack:
        raise ValueError("--save-batch-dir requires --binpack")
    if args.live_loader and (not args.binpack or args.save_batch_dir):
        raise ValueError("--live-loader requires --binpack and cannot save fixed batches")
    if args.repeat_live_batch and (not args.throughput_window or args.async_features):
        raise ValueError("repeating a live batch requires a non-async throughput window")
    if args.repeat_batch_cache and not args.repeat_live_batch:
        raise ValueError("--repeat-batch-cache requires --repeat-live-batch")
    if args.matched_batch_cache and (not args.repeat_live_batch or args.repeat_batch_cache):
        raise ValueError("--matched-batch-cache requires --repeat-live-batch and excludes --repeat-batch-cache")
    if args.matched_batch_index < 0 or (args.matched_batch_index and not args.matched_batch_cache):
        raise ValueError("matched-batch-index must be nonnegative and requires a matched cache")
    if args.matched_batch_count < 1 or (args.matched_batch_count > 1 and not args.matched_batch_cache):
        raise ValueError("matched-batch-count must be positive and multiple batches require a matched cache")
    if args.matched_batch_count > 1 and os.environ.get("NNUE_PROFILE_CACHE_FEATURES") == "1":
        raise ValueError("the single-batch feature cache is incompatible with rotating matched batches")
    if args.mode != "sharded" and (args.uniform_loss or args.overlap_replica_gradients or args.direct_packed_ft):
        raise ValueError("uniform loss, replica overlap and direct packed FT require sharded mode")
    if args.throughput_window and (not args.live_loader or args.profile):
        raise ValueError("--throughput-window requires --live-loader and excludes --profile")
    if args.measure_loader_wait and not args.throughput_window:
        raise ValueError("--measure-loader-wait requires --throughput-window")
    if args.measure_input_device_wait and not args.throughput_window:
        raise ValueError("--measure-input-device-wait requires --throughput-window")
    if args.nsys_capture and not args.throughput_window:
        raise ValueError("--nsys-capture requires --throughput-window")
    if args.async_features and (args.mode != "sharded" or not args.throughput_window
                                or args.measure_loader_wait or args.measure_input_device_wait
                                or os.environ.get("NNUE_BENCH_NATIVE_PACKED") != "1"):
        raise ValueError("async features require sharded native-packed live throughput mode without wait diagnostics")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    if args.global_batch % world:
        raise ValueError("global batch must be divisible by the process count")
    torch.set_num_threads((args.nettest_torch_threads or 1) if args.nettest_settings else 4)
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.nettest_settings:
        from nettest_benchmark_settings import make_nettest_settings

        from model.modules.layer_stacks import STACK_KEY_STATES, stack_count, stack_keys
        from model.modules.stack_table import count_stacks, fallback_table, table_to_str
        keys = stack_keys(ModelConfig().ls_keys)
        if args.nettest_remap_cache and os.path.isfile(args.nettest_remap_cache):
            cached = [None]
            if rank == 0:
                with open(args.nettest_remap_cache, encoding="utf-8") as cache:
                    cached[0] = cache.read().strip()
            dist.broadcast_object_list(cached, src=0)
            remap = cached[0]
        else:
            counts = count_stacks(args.nettest_remap_data, ModelConfig().ls_keys,
                                  stack_count(keys), args.loader_workers)
            total_counts = torch.from_numpy(counts).to(device=rank)
            dist.all_reduce(total_counts)
            remap = table_to_str(fallback_table(total_counts.cpu().numpy(), keys, STACK_KEY_STATES, 50))
            if args.nettest_remap_cache and rank == 0:
                with open(args.nettest_remap_cache, "x", encoding="utf-8") as cache:
                    cache.write(remap + "\n")
        config, loader_config, loader_io = make_nettest_settings(remap)
        if args.nettest_disable_batch_jitter:
            config.loss_params.lambda_config.jitter_lambda_batch = 0.0
        if args.nettest_ablate_filters:
            loader_config = DataloaderSkipConfig()
        if args.nettest_shuffle_entries is not None:
            loader_io.shuffle_buffer_entries = args.nettest_shuffle_entries
    else:
        config = NNUEConfig(
            model_config=ModelConfig(L1=1024, L2=32, L3=32),
            optimizer_config=OptimizerConfig(
                optimizer_name="adamw", adamw_fused=True, factorized_weight_decay=4e-5
            ),
        )
        loader_config = DataloaderSkipConfig()
        loader_io = DataloaderIOConfig(shuffle_buffer_entries=131072)
    seed = 1 if args.nettest_settings else 2026
    max_epoch = 560 if args.nettest_settings else 2
    num_batches = 1073741824 // args.global_batch if args.nettest_settings else 100
    torch.manual_seed(seed)
    nnue = NNUE(
        config,
        max_epoch=max_epoch,
        num_batches_per_epoch=num_batches,
        sharding=(rank, world) if args.mode == "sharded" else None,
    ).cuda(rank)
    if (args.matched_batch_cache or args.matched_initial_parameters) and args.mode == "sharded":
        match_sharded_initial_parameters(nnue, config, max_epoch=max_epoch,
                                         num_batches_per_epoch=num_batches, seed=seed)
    if args.ablate_shared_l1_factor:
        from experimental_no_l1_factor import apply_no_shared_l1_factor
        apply_no_shared_l1_factor(nnue)
    if os.environ.get("NNUE_BENCH_COMPILE_WEIGHT_CLIP") == "1":
        nnue.model._compile_layer_clipping = True
    live_loader = None
    ddp_prefetch = nnue.prefetch_hook() if args.mode == "ddp" else None
    native_requested = os.environ.get("NNUE_BENCH_NATIVE_PACKED") == "1"
    if args.direct_packed_ft and not native_requested:
        raise ValueError("--direct-packed-ft requires NNUE_BENCH_NATIVE_PACKED=1")
    if native_requested and args.mode != "sharded":
        raise ValueError("native packed benchmark input requires sharded mode")
    if args.live_loader:
        dataset = SparseBatchProvider(
            nnue.model.input_feature_name, args.binpack, args.global_batch // world,
            num_workers=args.loader_workers, config=loader_config,
            ddp_config=DataloaderDDPConfig(rank=rank, world_size=world),
            io_config=loader_io,
            use_pinned_memory=True,
            packed_sharded=native_requested and not args.matched_batch_cache,
        )
        live_loader = DevicePrefetcher(
            dataset, num_batches=args.warmup + args.steps + 2, queue_size_limit=2,
            device=torch.device("cuda", rank),
            prefetch_hook=ddp_prefetch if not args.repeat_live_batch else None,
        )
        cache_file = None
        replay_batches = ()
        if args.repeat_batch_cache:
            os.makedirs(args.repeat_batch_cache, exist_ok=True)
            layout = "packed" if native_requested else "dense"
            cache_file = os.path.join(args.repeat_batch_cache,
                                      f"batch-{args.mode}-{layout}-{rank}-{args.global_batch}.pt")
        if args.matched_batch_cache:
            replay_batches = load_matched_batches(
                live_loader, args.matched_batch_cache, args.matched_batch_index,
                args.matched_batch_count, rank, args.global_batch, torch.device("cuda", rank),
            )
            if any(batch[0].shape[0] != args.global_batch // world for batch in replay_batches):
                raise ValueError("matched batch cache does not match the local batch size")
            if native_requested:
                replay_batches = tuple(pack_saved_batch(batch) for batch in replay_batches)
            batch = replay_batches[0]
        elif cache_file and os.path.isfile(cache_file):
            batch = torch.load(cache_file, map_location=torch.device("cuda", rank), weights_only=True)
        else:
            batch = live_loader.next_batch()
            if cache_file:
                torch.cuda.synchronize()
                torch.save(batch, cache_file)
        if args.repeat_live_batch:
            replay_batches = replay_batches or (batch,)
            # End any cache-generation prefetch before warmup and timing.
            live_loader = dataset = None
        active = (batch[2]["nw"] / batch[2]["size"] if native_requested
                  else (batch[2] >= 0).sum().item() / batch[2].shape[0])
    elif args.load_batch_dir:
        path = os.path.join(args.load_batch_dir, f"batch-{rank}-{args.global_batch}.pt")
        batch = tuple(
            t.to(device=torch.device("cuda", rank))
            for t in torch.load(path, map_location="cpu", weights_only=True)
        )
        active = (batch[2] >= 0).sum().item() / batch[2].shape[0]
    elif args.binpack:
        provider = SparseBatchProvider(
            nnue.model.input_feature_name,
            args.binpack,
            args.global_batch // world,
            num_workers=args.loader_workers,
            config=loader_config,
            ddp_config=DataloaderDDPConfig(rank=rank, world_size=world),
            io_config=loader_io,
            device=torch.device("cuda", rank),
        )
        batch = next(provider)
        del provider
        active = (batch[2] >= 0).sum().item() / batch[2].shape[0]
        if args.save_batch_dir:
            os.makedirs(args.save_batch_dir, exist_ok=True)
            path = os.path.join(
                args.save_batch_dir, f"batch-{rank}-{args.global_batch}.pt"
            )
            torch.save(tuple(t.cpu() for t in batch), path)
    else:
        batch = make_batch(args.global_batch // world, torch.device("cuda", rank), rank)
        active = 64
    if native_requested and not args.live_loader:
        batch = pack_saved_batch(batch)
    def phase(name):
        if args.nsys_capture:
            return torch.cuda.nvtx.range(name)
        return torch.profiler.record_function(name) if args.profile else nullcontext()

    bucketed_bias_eligible = None
    if args.mode == "ddp":
        if args.compile_ddp:
            torch._dynamo.config.cache_size_limit = 64
            nnue.model = torch.compile(nnue.model, backend="inductor")
            nnue.enable_compiled_loss()
        optimizers, scheduler_specs = nnue.configure_optimizers()
        optimizer = optimizers[0]
        schedulers = [s.get("scheduler", s) if isinstance(s, dict) else s for s in scheduler_specs]
        ddp_step = 0
        if args.ddp_freeze_weights:
            for group in optimizer.param_groups:
                group["lr"] = 0.0
        ddp = DDP(
            nnue.model,
            device_ids=[rank],
            gradient_as_bucket_view=True,
            static_graph=True,
            find_unused_parameters=False,
            broadcast_buffers=False,
            bucket_cap_mb=50,
        )
        bucket_info = {}
        if args.ddp_gradient_mode != "normal":
            from ddp_comm_diagnostics import make_diagnostic_hook
            ddp.register_comm_hook(dist.group.WORLD,
                                   make_diagnostic_hook(nnue.model, args.ddp_gradient_mode, bucket_info))

        def train_step():
            nonlocal ddp_step
            bucket_info.clear()
            optimizer.zero_grad(set_to_none=True)
            if ddp_prefetch is not None and live_loader is None:
                # Replay excludes loader work, but still pays tiled FT preparation.
                with phase("step:ddp_tiled_prep"):
                    ddp_prefetch(batch)
            us, them, white, black, _, _, stacks = batch
            with phase("step:ddp_forward"):
                output = ddp(us, them, white, black, stacks, True, True)
            with phase("step:ddp_loss"):
                loss = nnue.compute_loss_with_scorenet(output, batch, ddp_step if args.nettest_settings else 0)
            with phase("step:ddp_backward"):
                loss.backward()
            with phase("step:ddp_clip"):
                if not args.ablate_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(nnue.model.parameters(), 2.0)
            with phase("step:ddp_optimizer"):
                optimizer.step()
            if args.nettest_settings:
                from trainer.engine import is_one_cycle
                with phase("step:ddp_scheduler"):
                    for scheduler in schedulers:
                        if is_one_cycle(scheduler):
                            scheduler.step()
                if not args.nettest_ablate_weight_clipping:
                    with phase("step:ddp_weight_clip"):
                        nnue.model.clip_weights(include_input=False)
            ddp_step += 1
            return loss.detach()
    else:
        if profile_compile and os.environ.get("NNUE_PROFILE_WRAPPER") != "1":
            if os.environ.get("NNUE_PROFILE_COMPILE_FT_MERGE") == "1":
                nnue.model.input._merge_weight_and_bias_torch = torch.compile(
                    nnue.model.input._merge_weight_and_bias_torch, fullgraph=True
                )
            if os.environ.get("NNUE_PROFILE_COMPILE_TAIL") == "1":
                nnue.model.layer_stacks.forward_from_l1 = torch.compile(
                    nnue.model.layer_stacks.forward_from_l1
                )
        if os.environ.get("NNUE_PROFILE_COMPILE_LOSS") == "1":
            nnue.enable_compiled_loss()
        runner = ShardedAdamWStep(
            nnue, gradient_clip_val=None if args.ablate_gradient_clipping else 2.0,
            compile_model=False if args.eager_sharded or profile_compile else None,
            bucketed_bias=args.bucketed_l1_bias,
            uniform_loss=args.uniform_loss,
            overlap_replica_gradients=args.overlap_replica_gradients,
            direct_packed_ft=args.direct_packed_ft,
        )
        if os.environ.get("NNUE_BENCH_REFERENCE_LAUNCHES") == "1":
            nnue.model._compile_layer_clipping = False
            nnue.model.layer_stacks.l1._compile_l1_merge = False
        if args.ablate_shared_l1_factor:
            runner.sharded = [p for p in runner.sharded if p.requires_grad]
            runner.replicated = [p for p in runner.replicated if p.requires_grad]
        if args.bucketed_l1_bias:
            from model.modules.bucketed_bias import _eligible
            l1 = nnue.model.layer_stacks.l1
            local_indices = batch[-1]
            probe = torch.empty((local_indices.numel(), l1.out_features),
                                dtype=torch.float32, device=local_indices.device)
            bucketed_bias_eligible = _eligible(probe, l1.linear.bias, local_indices,
                                              l1.count, l1.out_features)
            del probe
            if args.require_bucketed_l1_fastpath and not bucketed_bias_eligible:
                raise RuntimeError("bucketed L1 bias fast path is not eligible for this batch")
        if args.nsys_capture:
            def mark_method(obj, name, label):
                original = getattr(obj, name)
                @functools.wraps(original)
                def marked(*call_args, **call_kwargs):
                    with phase(label):
                        return original(*call_args, **call_kwargs)
                setattr(obj, name, marked)

            mark_method(nnue.model.input, "merged_weight_and_bias", "step:sharded_ft_merge")
            mark_method(nnue.model.layer_stacks, "forward_from_l1", "step:sharded_tail")
            mark_method(nnue, "compute_loss_components_with_scorenet", "step:sharded_loss")

        def train_step(global_batch=None, after_forward=None):
            with phase("step:sharded_train"):
                result = runner.step(batch, global_batch=global_batch, after_forward=after_forward)
            if args.nettest_settings and not args.nettest_ablate_weight_clipping:
                with phase("step:sharded_weight_clip"):
                    nnue.model.clip_weights(include_input=False)
            return result

    if args.throughput_window:
        pipeline_batch = pipeline_pending = feature_stream = None
        total_pipeline_steps = args.warmup + args.steps
        if args.async_features:
            from data_loader.dataset import _recursive_record_stream
            from data_loader.distributed_features import begin_native_packed_features
            feature_stream = torch.cuda.Stream(device=rank)

            def prepare(index):
                nvtx = (torch.cuda.nvtx.range("phase:async_feature_prepare")
                        if os.environ.get("NNUE_PROFILE_NVTX") == "1" else nullcontext())
                with nvtx, torch.cuda.stream(feature_stream):
                    local = live_loader.next_batch()
                    pending = begin_native_packed_features(local)
                return local, pending

            pipeline_batch, pipeline_pending = prepare(1)

        def one_step(index):
            nonlocal batch, pipeline_batch, pipeline_pending
            if not args.async_features:
                with phase("step:loader_acquire"):
                    batch = (replay_batches[index % len(replay_batches)] if replay_batches
                             else live_loader.next_batch())
                return train_step()
            batch = pipeline_batch
            nvtx = (torch.cuda.nvtx.range("phase:async_feature_finish")
                    if os.environ.get("NNUE_PROFILE_NVTX") == "1" else nullcontext())
            with nvtx:
                features = pipeline_pending.finish(
                    direct_packed_ft=nnue.model.can_use_direct_packed_ft()
                )
            _recursive_record_stream(batch, torch.cuda.current_stream(rank))
            next_batch = next_pending = None

            def start_next():
                nonlocal next_batch, next_pending
                if index + 1 < total_pipeline_steps:
                    next_batch, next_pending = prepare(index + 2)

            loss = train_step(features, start_next)
            pipeline_batch, pipeline_pending = next_batch, next_pending
            return loss

        for i in range(args.warmup):
            last_loss = one_step(i)
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        if args.nsys_capture:
            torch.cuda.cudart().cudaProfilerStart()
        loader_stats_start = (
            live_loader.get_loader_stats()
            if args.measure_loader_wait and live_loader is not None else None
        )
        started = time.perf_counter()
        host_loader_wait = []
        input_device_events = []
        for i in range(args.steps):
            if args.async_features:
                last_loss = one_step(args.warmup + i)
                continue
            if args.measure_input_device_wait:
                before = torch.cuda.Event(enable_timing=True)
                after = torch.cuda.Event(enable_timing=True)
                before.record()
            if args.measure_loader_wait:
                wait_started = time.perf_counter()
            with phase("step:loader_acquire"):
                batch = (replay_batches[(args.warmup + i) % len(replay_batches)] if replay_batches
                         else live_loader.next_batch())
            if args.measure_loader_wait:
                host_loader_wait.append(time.perf_counter() - wait_started)
            if args.measure_input_device_wait:
                after.record()
                input_device_events.append((before, after))
            last_loss = train_step()
        torch.cuda.synchronize()
        finished = time.perf_counter()
        loader_stats_end = live_loader.get_loader_stats() if loader_stats_start is not None else None
        if args.nsys_capture:
            torch.cuda.cudart().cudaProfilerStop()
        elapsed = torch.tensor(finished - started, device=rank)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        peak = torch.tensor(torch.cuda.max_memory_allocated(), dtype=torch.int64, device=rank)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        finite = torch.isfinite(last_loss).to(torch.int32)
        for parameter in nnue.model.parameters():
            finite &= torch.isfinite(parameter).all().to(torch.int32)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not bool(finite):
            raise RuntimeError("benchmark produced a non-finite loss or parameter")
        wait_by_rank = None
        if args.measure_loader_wait:
            ordered = sorted(host_loader_wait)
            local_wait = {"rank": rank, "total_seconds": sum(ordered),
                          "median_ms": statistics.median(ordered) * 1000,
                          "p95_ms": ordered[int(0.95 * (len(ordered) - 1))] * 1000,
                          "max_ms": ordered[-1] * 1000}
            if loader_stats_end is not None:
                local_wait["pipeline_delta"] = {
                    key: loader_stats_end[key] - loader_stats_start[key]
                    for key in loader_stats_end
                }
            wait_by_rank = [None] * world
            dist.all_gather_object(wait_by_rank, local_wait)
        device_wait_by_rank = None
        if args.measure_input_device_wait:
            ordered_ms = sorted(before.elapsed_time(after) for before, after in input_device_events)
            local_device_wait = {"rank": rank, "total_ms": sum(ordered_ms),
                                 "median_ms": statistics.median(ordered_ms),
                                 "p95_ms": ordered_ms[int(0.95 * (len(ordered_ms) - 1))],
                                 "max_ms": ordered_ms[-1]}
            device_wait_by_rank = [None] * world
            dist.all_gather_object(device_wait_by_rank, local_device_wait)
        if rank == 0:
            print(json.dumps({"mode": args.mode, "world_size": world,
                              "global_batch": args.global_batch, "steps": args.steps,
                              "live_loader": True, "throughput_window": True,
                              "repeat_live_batch": args.repeat_live_batch,
                              "matched_batch": bool(args.matched_batch_cache),
                              "matched_batch_index": args.matched_batch_index if args.matched_batch_cache else None,
                              "matched_batch_count": args.matched_batch_count if args.matched_batch_cache else None,
                              "matched_initial_parameters": bool(args.matched_batch_cache or args.matched_initial_parameters),
                              "nettest_settings": args.nettest_settings,
                              "nettest_batch_jitter_disabled": args.nettest_disable_batch_jitter,
                              "batch_lambda_jitter_std": config.loss_params.lambda_config.jitter_lambda_batch,
                              "nettest_quantization_phase": "post_warmup" if args.nettest_settings else None,
                              "features": config.features,
                              "nettest_filter_ablation": args.nettest_ablate_filters,
                              "nettest_weight_clipping_ablation": args.nettest_ablate_weight_clipping,
                              "gradient_clipping_ablation": args.ablate_gradient_clipping,
                              "shared_l1_factor_ablation": args.ablate_shared_l1_factor,
                              "torch_threads_per_rank": torch.get_num_threads(),
                              "shuffle_buffer_entries": loader_io.shuffle_buffer_entries,
                              "loader_workers_per_rank": args.loader_workers,
                              "stored_layer_stacks": nnue.model.layer_stacks.count,
                              "async_features": args.async_features,
                              "sharded_experiments": runner.last_experiments if args.mode == "sharded" else None,
                              "ddp_tiled_prep": (
                                  "prefetch" if live_loader is not None else "inline"
                              ) if ddp_prefetch is not None else None,
                              "bucketed_l1_bias": (args.bucketed_l1_bias
                                                   or os.environ.get("NNUE_PROFILE_BUCKETED_BIAS") == "1"),
                              "bucketed_l1_fastpath_eligible": bucketed_bias_eligible,
                              "compiled_layer_clipping": getattr(nnue.model, "_compile_layer_clipping", False),
                              "compiled_l1_merge": (getattr(nnue.model.layer_stacks.l1, "_compile_l1_merge", False)
                                                    if args.mode == "sharded" else None),
                              "reference_l1_collective": os.environ.get("NNUE_PROFILE_REFERENCE_L1_COLLECTIVE") == "1",
                              "compiled": (args.compile_ddp if args.mode == "ddp" else
                                           runner.compile_model or profile_compile),
                              "gpu": torch.cuda.get_device_name(rank),
                              "compiled_sharded_ft_merge": (runner.compile_model or os.environ.get("NNUE_PROFILE_COMPILE_FT_MERGE") == "1") if args.mode == "sharded" else None,
                              "compiled_sharded_tail": (runner.compile_model or os.environ.get("NNUE_PROFILE_COMPILE_TAIL") == "1") if args.mode == "sharded" else None,
                              "compiled_sharded_loss": (runner.compile_model or os.environ.get("NNUE_PROFILE_COMPILE_LOSS") == "1") if args.mode == "sharded" else None,
                              "elapsed_seconds": elapsed.item(),
                              "positions_per_second": args.global_batch * args.steps / elapsed.item(),
                              "host_loader_wait_by_rank": wait_by_rank,
                              "input_device_interval_by_rank": device_wait_by_rank,
                              "peak_allocated_gib": peak.item() / 2**30}), flush=True)
        dist.destroy_process_group()
        return

    times, loader_wait_times, training_times = [], [], []
    torch.cuda.reset_peak_memory_stats()
    for iteration in range(args.warmup + args.steps):
        dist.barrier()
        torch.cuda.synchronize()
        start = time.perf_counter()
        if live_loader is not None:
            batch = live_loader.next_batch()
        input_done = time.perf_counter()
        last_loss = train_step()
        torch.cuda.synchronize()
        end = time.perf_counter()
        elapsed = torch.tensor([end - start, input_done - start, end - input_done], device=rank)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        if iteration >= args.warmup:
            times.append(elapsed[0].item())
            loader_wait_times.append(elapsed[1].item())
            training_times.append(elapsed[2].item())
    peak = torch.tensor(
        torch.cuda.max_memory_allocated(), dtype=torch.int64, device=rank
    )
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    finite = torch.isfinite(last_loss).to(torch.int32)
    for parameter in nnue.model.parameters():
        finite &= torch.isfinite(parameter).all().to(torch.int32)
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not bool(finite):
        raise RuntimeError("benchmark produced a non-finite loss or parameter")
    if rank == 0:
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "world_size": world,
                    "global_batch": args.global_batch,
                    "batch_lambda_jitter_std": config.loss_params.lambda_config.jitter_lambda_batch,
                    "sharded_experiments": runner.last_experiments if args.mode == "sharded" else None,
                    "mean_white_active_rank0": active,
                    "input": "binpack"
                    if args.binpack or args.load_batch_dir
                    else "synthetic",
                    "live_loader": args.live_loader,
                    "loader_wait_seconds": loader_wait_times if args.live_loader else None,
                    "training_seconds": training_times if args.live_loader else None,
                    "compiled": (args.compile_ddp if args.mode == "ddp" else
                                 runner.compile_model or profile_compile),
                    "compiled_sharded_ft_merge": (runner.compile_model or os.environ.get("NNUE_PROFILE_COMPILE_FT_MERGE") == "1") if args.mode == "sharded" else None,
                    "compiled_sharded_tail": (runner.compile_model or os.environ.get("NNUE_PROFILE_COMPILE_TAIL") == "1") if args.mode == "sharded" else None,
                    "compiled_sharded_loss": (runner.compile_model or os.environ.get("NNUE_PROFILE_COMPILE_LOSS") == "1") if args.mode == "sharded" else None,
                    "ddp_gradient_mode": args.ddp_gradient_mode if args.mode == "ddp" else None,
                    "ddp_freeze_weights": args.ddp_freeze_weights if args.mode == "ddp" else None,
                    "ddp_buckets": list(bucket_info.values()) if args.mode == "ddp" else None,
                    "gpu": torch.cuda.get_device_name(rank),
                    "step_seconds": times,
                    "median_seconds": statistics.median(times),
                    "positions_per_second": args.global_batch
                    / statistics.median(times),
                    "peak_allocated_gib": peak.item() / 2**30,
                    "tf32": True,
                }
            ),
            flush=True,
        )
    if args.profile:
        dist.barrier()
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as profile:
            if args.profile_all_ranks:
                # Profiler setup can take different amounts of host time per
                # rank. Synchronize only after every profiler is active so
                # collective kernels do not mostly measure startup skew.
                dist.barrier()
                torch.cuda.synchronize()
            train_step()
            torch.cuda.synchronize()
        if rank == 0:
            phase_events = [event for event in profile.key_averages()
                            if event.key.startswith("phase:")]
            if phase_events:
                print("PROFILE_PHASES " + json.dumps([
                    {"name": event.key, "calls": event.count,
                     "cpu_total_ms": event.cpu_time_total / 1000,
                     "device_total_ms": event.device_time_total / 1000}
                    for event in phase_events
                ]), flush=True)
            print(
                profile.key_averages(group_by_input_shape=True).table(
                    sort_by="self_device_time_total", row_limit=25
                ),
                flush=True,
            )
        if rank == 0 or args.profile_all_ranks:
            suffix = f"-rank{rank}" if args.profile_all_ranks else ""
            mode_label = args.ddp_gradient_mode if args.mode == "ddp" else "sharded"
            profile.export_chrome_trace(
                f"profile-{args.mode}-{mode_label}-{args.global_batch}{suffix}.json"
            )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

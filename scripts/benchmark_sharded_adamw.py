"""Time eager DDP versus FT/L1-sharded AdamW with synthetic sparse features.

Run separately for each mode with torchrun --standalone --nproc-per-node=2.
Times exclude data loading and initialization and include forward, backward,
gradient synchronization, clipping, and the optimizer update.
"""

import argparse
from contextlib import nullcontext
import json
import os
import statistics
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data_loader import DataloaderSkipConfig, DevicePrefetcher, SparseBatchProvider
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("ddp", "sharded"))
    parser.add_argument("--global-batch", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--binpack", help="Use one native-loader batch rather than synthetic features"
    )
    parser.add_argument("--loader-workers", type=int, default=8)
    parser.add_argument("--live-loader", action="store_true",
                        help="Include asynchronous binpack loading in each timed step")
    parser.add_argument("--throughput-window", action="store_true",
                        help="Time a sustained live-loader window without per-step barriers")
    parser.add_argument("--measure-loader-wait", action="store_true",
                        help="Record host-side batch acquisition time during throughput windows")
    parser.add_argument("--measure-input-device-wait", action="store_true",
                        help="Measure the GPU stream interval surrounding batch acquisition")
    parser.add_argument(
        "--save-batch-dir", help="Save decoded rank batches for an identical comparison"
    )
    parser.add_argument("--load-batch-dir", help="Reuse previously saved rank batches")
    parser.add_argument(
        "--compile-ddp",
        action="store_true",
        help="Compile the DDP model and loss as train.py does",
    )
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
    if args.compile_ddp and args.mode != "ddp":
        raise ValueError("--compile-ddp applies only to the DDP baseline")
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
    if args.throughput_window and (not args.live_loader or args.profile):
        raise ValueError("--throughput-window requires --live-loader and excludes --profile")
    if args.measure_loader_wait and not args.throughput_window:
        raise ValueError("--measure-loader-wait requires --throughput-window")
    if args.measure_input_device_wait and not args.throughput_window:
        raise ValueError("--measure-input-device-wait requires --throughput-window")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    if args.global_batch % world:
        raise ValueError("global batch must be divisible by the process count")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    config = NNUEConfig(
        model_config=ModelConfig(L1=1024, L2=32, L3=32),
        optimizer_config=OptimizerConfig(
            optimizer_name="adamw", adamw_fused=True, factorized_weight_decay=4e-5
        ),
    )
    torch.manual_seed(2026)
    nnue = NNUE(
        config,
        max_epoch=2,
        num_batches_per_epoch=100,
        sharding=(rank, world) if args.mode == "sharded" else None,
    ).cuda(rank)
    live_loader = None
    native_requested = os.environ.get("NNUE_BENCH_NATIVE_PACKED") == "1"
    if native_requested and args.mode != "sharded":
        raise ValueError("native packed benchmark input requires sharded mode")
    if args.live_loader:
        dataset = SparseBatchProvider(
            nnue.model.input_feature_name, [args.binpack], args.global_batch // world,
            num_workers=args.loader_workers, config=DataloaderSkipConfig(),
            ddp_config=DataloaderDDPConfig(rank=rank, world_size=world),
            io_config=DataloaderIOConfig(shuffle_buffer_entries=131072),
            use_pinned_memory=True,
            packed_sharded=native_requested,
        )
        live_loader = DevicePrefetcher(dataset, num_batches=args.warmup + args.steps + 2,
                                       queue_size_limit=2, device=torch.device("cuda", rank))
        batch = live_loader.next_batch()
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
            [args.binpack],
            args.global_batch // world,
            num_workers=args.loader_workers,
            config=DataloaderSkipConfig(),
            ddp_config=DataloaderDDPConfig(rank=rank, world_size=world),
            io_config=DataloaderIOConfig(shuffle_buffer_entries=131072),
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
    if args.mode == "ddp":
        if args.compile_ddp:
            torch._dynamo.config.cache_size_limit = 64
            nnue.model = torch.compile(nnue.model, backend="inductor")
            nnue.enable_compiled_loss()
        optimizer = nnue.configure_optimizers()[0][0]
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

        def phase(name):
            return torch.profiler.record_function(name) if args.profile else nullcontext()

        def train_step():
            bucket_info.clear()
            optimizer.zero_grad(set_to_none=True)
            us, them, white, black, _, _, stacks = batch
            with phase("step:ddp_forward"):
                output = ddp(us, them, white, black, stacks, True, True)
                loss = nnue.compute_loss_with_scorenet(output, batch, 0)
            with phase("step:ddp_backward"):
                loss.backward()
            with phase("step:ddp_clip"):
                torch.nn.utils.clip_grad_norm_(nnue.model.parameters(), 2.0)
            with phase("step:ddp_optimizer"):
                optimizer.step()
            return loss.detach()
    else:
        runner = ShardedAdamWStep(nnue, gradient_clip_val=2.0)

        def train_step():
            return runner.step(batch)

    if args.throughput_window:
        for i in range(args.warmup):
            batch = live_loader.next_batch()
            last_loss = train_step()
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        host_loader_wait = []
        input_device_events = []
        for i in range(args.steps):
            if args.measure_input_device_wait:
                before = torch.cuda.Event(enable_timing=True)
                after = torch.cuda.Event(enable_timing=True)
                before.record()
            if args.measure_loader_wait:
                wait_started = time.perf_counter()
            batch = live_loader.next_batch()
            if args.measure_loader_wait:
                host_loader_wait.append(time.perf_counter() - wait_started)
            if args.measure_input_device_wait:
                after.record()
                input_device_events.append((before, after))
            last_loss = train_step()
        torch.cuda.synchronize()
        elapsed = torch.tensor(time.perf_counter() - started, device=rank)
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
                              "compiled": args.compile_ddp, "gpu": torch.cuda.get_device_name(rank),
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
                    "mean_white_active_rank0": active,
                    "input": "binpack"
                    if args.binpack or args.load_batch_dir
                    else "synthetic",
                    "live_loader": args.live_loader,
                    "loader_wait_seconds": loader_wait_times if args.live_loader else None,
                    "training_seconds": training_times if args.live_loader else None,
                    "compiled": args.compile_ddp,
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

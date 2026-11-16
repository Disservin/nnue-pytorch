"""Benchmark native feature extraction from one reusable compact binpack batch.

Example: python scripts/benchmark_compact_features.py data.binpack --workers 1 2 4 8
Reports native allocation + extraction time; excludes file decoding, Python
tensor copies, device transfer and freeing the output batch.
"""

import argparse
import ctypes
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loader import CompactBatchStream, DataloaderIOConfig, DataloaderSkipConfig
from data_loader._native import NativeSparseBatchPointer, c_lib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binpacks", nargs="+")
    parser.add_argument("--batch-size", type=int, default=131072)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--backends", nargs="+", choices=["direct", "native"],
                        default=["native", "direct"])
    parser.add_argument("--loader-workers", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    if min(args.batch_size, args.loader_workers, args.iterations, *args.workers) < 1 or args.warmup < 0:
        parser.error("batch sizes, workers, and iterations must be positive")

    # No stochastic filtering: this microbenchmark repeatedly uses one batch.
    config = DataloaderSkipConfig(
        filtered=False, wld_filtered=False, early_fen_skipping=-1,
        soft_early_fen_skipping=0, random_fen_skipping=0,
    )
    io_config = DataloaderIOConfig(shuffle_buffer_entries=max(1024, min(args.batch_size, 1 << 20)))
    with CompactBatchStream(args.binpacks, args.batch_size, num_workers=args.loader_workers,
                            cyclic=True, config=config, io_config=io_config) as stream:
        records = next(stream).records

    lib = c_lib.dll
    fn = lib.get_sparse_batch_from_compact_backend
    fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint8),
                   ctypes.c_size_t, ctypes.c_int, ctypes.c_bool,
                   ctypes.POINTER(ctypes.c_char), ctypes.c_size_t]
    fn.restype = NativeSparseBatchPointer
    print(f"batch={len(records)}; native extraction only; median of {args.iterations} runs")
    for workers in args.workers:
        samples = {backend: [] for backend in args.backends}
        for i in range(args.warmup + args.iterations):
            order = args.backends if i % 2 == 0 else list(reversed(args.backends))
            for backend in order:
                error = ctypes.create_string_buffer(512)
                started = time.perf_counter()
                batch = fn(b"Full_Threats+PP_3Wide+K32H3", b"piececount:queen:pawnhalf:manyoffi",
                           records.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), records.nbytes,
                           workers, backend == "direct", error, len(error))
                elapsed = time.perf_counter() - started
                if not batch:
                    raise RuntimeError(error.value.decode("utf-8", errors="replace"))
                lib.destroy_sparse_batch(batch)
                if i >= args.warmup:
                    samples[backend].append(elapsed)
        for backend in args.backends:
            median = statistics.median(samples[backend])
            print(f"workers={workers} backend={backend}: {median * 1000:.2f} ms; "
                  f"{len(records) / median / 1e6:.3f} M positions/s")


if __name__ == "__main__":
    main()

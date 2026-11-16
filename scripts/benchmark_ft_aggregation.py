"""Compare FT backward kernels using the saved real-feature benchmark batches."""

import argparse
import json
import os
import statistics
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.modules.feature_transformer.aggregated_ft_kernel import aggregated_ft_backward
from model.modules.feature_transformer.fused_ft_kernel import (
    BACKWARD_TILE_SIZE,
    make_fused_double_ft_backward_kernel,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--batch", type=int, default=131072)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--tile", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    torch.manual_seed(12345)
    parts = [torch.load(os.path.join(args.batch_dir, f"batch-{r}-{args.batch}.pt"),
                        map_location="cpu", weights_only=True) for r in (0, 1)]
    us, them, white, black = [torch.cat([p[i] for p in parts], 0).cuda().contiguous()
                              for i in range(4)]
    batch, active = white.shape
    assert batch == args.batch
    rows = int(torch.maximum(white.max(), black.max()).item()) + 1
    width = args.width
    grad = torch.randn(batch, width, device="cuda") / batch**0.5
    clamped = torch.rand(batch, 4, width // 2, device="cuda")
    clamped[clamped < .2] = 0
    clamped[clamped > .8] = 1
    gw = torch.zeros(rows, width, device="cuda")
    gb = torch.zeros(width, device="cuda")
    direct = make_fused_double_ft_backward_kernel(active, width)

    def run_direct():
        direct(grid=((batch + BACKWARD_TILE_SIZE - 1) // BACKWARD_TILE_SIZE,), args=(
            us.data_ptr(), them.data_ptr(), white.data_ptr(), black.data_ptr(),
            gw.data_ptr(), gb.data_ptr(), np.float32(1), grad.data_ptr(),
            clamped.data_ptr(), gw.data_ptr(), gb.data_ptr(), np.int32(batch), np.int32(width)))

    def run_aggregate():
        aggregated_ft_backward(us, them, white, black, grad, clamped, gw, gb, 1.,
                               tile_size=args.tile)

    def reset():
        gw.zero_()
        gb.zero_()

    reset(); run_direct(); torch.cuda.synchronize()
    reference = gw.clone()
    reference_bias = gb.clone()
    reset(); run_aggregate(); torch.cuda.synchronize()
    abs_error = (reference - gw).abs().max().item()
    bias_error = (reference_bias - gb).abs().max().item()
    torch.testing.assert_close(gw, reference, atol=2e-4, rtol=2e-3)
    torch.testing.assert_close(gb, reference_bias, atol=2e-4, rtol=2e-3)
    timings = {}
    for name, fn in (("direct", run_direct), ("aggregated", run_aggregate)):
        for _ in range(3):
            reset(); fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.steps):
            reset()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); fn(); end.record(); end.synchronize()
            samples.append(start.elapsed_time(end))
        timings[name] = {"median_ms": statistics.median(samples), "samples_ms": samples}
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "batch": batch, "width": width,
                      "tile": args.tile, "active_capacity": active, "feature_rows": rows,
                      "max_weight_error": abs_error, "max_bias_error": bias_error,
                      "timings": timings}), flush=True)


if __name__ == "__main__":
    main()

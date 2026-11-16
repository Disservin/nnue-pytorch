"""Run the sharded-step benchmark with an experimental FT backward choice.

FT_BENCH_MODE=direct or aggregated is required. This deliberately allows
256-wide aggregation for benchmarking; it does not change production defaults.
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.modules.feature_transformer import fused_ft_functions as ft

mode = os.environ["FT_BENCH_MODE"]
if mode not in ("direct", "aggregated"):
    raise ValueError("FT_BENCH_MODE must be direct or aggregated")


def experimental_selector(width, batch, active, device):
    if mode == "direct":
        return False
    capability = ft.torch.cuda.get_device_capability(device)
    return (ft.torch.version.hip is None and capability[0] >= 8 and
            256 <= width <= 4096 and width % 128 == 0 and 0 < active <= 288)


ft._use_aggregated_backward = experimental_selector
if "FT_BENCH_TILE" in os.environ:
    tile = int(os.environ["FT_BENCH_TILE"])
    if not 1 <= tile <= 16:
        raise ValueError("FT_BENCH_TILE must be between 1 and 16")
    ft._aggregated_tile_size = lambda width, batch, active, capability: tile
runpy.run_path(os.path.join(os.path.dirname(__file__), "benchmark_sharded_adamw.py"),
               run_name="__main__")

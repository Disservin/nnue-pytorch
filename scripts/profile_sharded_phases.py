"""Run benchmark_sharded_adamw with labeled PyTorch profiler regions."""
import functools
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import data_loader.distributed_features as features
import model.sharded_backward as backward
import model.sharded_model as model
import trainer.sharded_adamw as trainer
import model.modules.sharded_l1 as sharded_l1
import model.modules.feature_transformer.fused_ft_functions as ft_functions
from model.modules.feature_transformer.composed_feature_transformer import ComposedFeatureTransformer
from model.modules.layer_stacks import LayerStacks
from model.modules.sharded_l1 import ShardedL1


def wrap(obj, name, label):
    if os.environ.get("NNUE_PROFILE_NO_RANGES") == "1":
        return
    original = getattr(obj, name)

    @functools.wraps(original)
    def measured(*args, **kwargs):
        with torch.profiler.record_function(label):
            return original(*args, **kwargs)
    setattr(obj, name, measured)


if os.environ.get("NNUE_PROFILE_REFERENCE_UNPACK") == "1":
    features._unpack_rows = features._unpack_rows_reference
    features._use_fused_unpack = lambda device: False
if os.environ.get("NNUE_PROFILE_SEPARATE_UNPACK") == "1":
    features._use_fused_unpack = lambda device: False
if os.environ.get("NNUE_PROFILE_COMPILE_FT_MERGE") == "1":
    ComposedFeatureTransformer.merged_weight_and_bias = torch.compile(
        ComposedFeatureTransformer.merged_weight_and_bias, fullgraph=True
    )
if os.environ.get("NNUE_PROFILE_BUCKETED_BIAS") == "1":
    from experimental_bias_add import bucketed_bias_add
    sharded_l1._add_local_bias = bucketed_bias_add
if os.environ.get("NNUE_PROFILE_FT_BACKWARD") in ("direct", "aggregated"):
    mode = os.environ["NNUE_PROFILE_FT_BACKWARD"]
    def ft_selector(width, batch, active, device):
        return (mode == "aggregated" and torch.version.hip is None
                and torch.cuda.get_device_capability(device)[0] >= 8
                and 256 <= width <= 4096 and width % 128 == 0
                and 0 < active <= 288)
    ft_functions._use_aggregated_backward = ft_selector
    if "NNUE_PROFILE_FT_TILE" in os.environ:
        tile = int(os.environ["NNUE_PROFILE_FT_TILE"])
        if not 1 <= tile <= 16:
            raise ValueError("NNUE_PROFILE_FT_TILE must be between 1 and 16")
        ft_functions._aggregated_tile_size = lambda *args: tile
wrap(model, "gather_packed_features", "phase:feature_gather_total")
wrap(features, "_unpack_rows", "phase:feature_unpack")
wrap(ComposedFeatureTransformer, "merged_weight_and_bias", "phase:ft_merge_quantize")
wrap(ComposedFeatureTransformer, "forward", "phase:ft_forward_total")
wrap(ShardedL1, "forward", "phase:sharded_l1_forward")
wrap(LayerStacks, "forward_from_l1", "phase:tail_forward")
wrap(backward, "global_weighted_loss", "phase:loss_denominator")
wrap(backward, "synchronize_replicated_gradients", "phase:replica_grad_sync")
wrap(trainer, "backward_weighted_loss", "phase:backward_total")
wrap(trainer, "clip_sharded_grad_norm_", "phase:clip_total")

# Measurement-only upper bound for eliminating feature distribution. The
# benchmark must reuse one fixed batch and all ranks must enter the first call.
if os.environ.get("NNUE_PROFILE_CACHE_FEATURES") == "1":
    original_gather = model.gather_packed_features
    cache = []

    def cached_gather(batch, group=None):
        if not cache:
            cache.append(original_gather(batch, group))
        return cache[0]

    model.gather_packed_features = cached_gather

# Measurement-only upper bound: this changes the optimizer's input gradients.
if os.environ.get("NNUE_PROFILE_SKIP_CLIP") == "1":
    trainer.clip_sharded_grad_norm_ = lambda *args, **kwargs: None

runpy.run_path(os.path.join(os.path.dirname(__file__), "benchmark_sharded_adamw.py"),
               run_name="__main__")

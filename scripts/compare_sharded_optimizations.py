"""Compare baseline and experimental sharded steps from identical parameters."""
import argparse
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_loader.distributed_features as features
import model.modules.feature_transformer.fused_ft_functions as ft
import model.modules.sharded_l1 as l1
from model.config import ModelConfig, NNUEConfig
from model.modules.feature_transformer.composed_feature_transformer import ComposedFeatureTransformer
from model.nnue import NNUE
from model.optimizers import OptimizerConfig
from trainer.sharded_adamw import ShardedAdamWStep
from experimental_bias_add import bucketed_bias_add
from benchmark_sharded_adamw import pack_saved_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--batch", type=int, default=131072)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--native-packed", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg = NNUEConfig(model_config=ModelConfig(L1=1024, L2=32, L3=32),
                     optimizer_config=OptimizerConfig(optimizer_name="adamw", adamw_fused=True,
                                                       factorized_weight_decay=4e-5))
    torch.manual_seed(2026)
    baseline = NNUE(cfg, max_epoch=2, num_batches_per_epoch=100, sharding=(rank, world)).cuda(rank)
    torch.manual_seed(2026)
    candidate = NNUE(cfg, max_epoch=2, num_batches_per_epoch=100, sharding=(rank, world)).cuda(rank)
    candidate.load_state_dict(baseline.state_dict())
    runner0 = ShardedAdamWStep(baseline, gradient_clip_val=2.0, compile_model=False)
    runner1 = ShardedAdamWStep(candidate, gradient_clip_val=2.0, compile_model=False)
    batch = tuple(t.cuda(rank) for t in torch.load(
        os.path.join(args.batch_dir, f"batch-{rank}-{args.batch}.pt"),
        map_location="cpu", weights_only=True))
    candidate_batch = pack_saved_batch(batch) if args.native_packed else batch
    selector, backward, tile, merge, bias = (features._use_fused_unpack,
        ft._use_aggregated_backward, ft._aggregated_tile_size,
        ComposedFeatureTransformer.merged_weight_and_bias, l1._add_local_bias)
    compiled_merge = torch.compile(merge, fullgraph=True)
    for i in range(args.steps):
        features._use_fused_unpack = lambda device: False
        ft._use_aggregated_backward = lambda *a, **k: False
        ComposedFeatureTransformer.merged_weight_and_bias = merge
        l1._add_local_bias = bias
        expected_loss = runner0.step(batch)
        features._use_fused_unpack = selector
        ft._use_aggregated_backward = lambda *a, **k: True
        ft._aggregated_tile_size = lambda *a, **k: 14
        ComposedFeatureTransformer.merged_weight_and_bias = compiled_merge
        l1._add_local_bias = bucketed_bias_add
        actual_loss = runner1.step(candidate_batch)
        torch.testing.assert_close(actual_loss, expected_loss, atol=2e-5, rtol=2e-4)
        for (name, ref), (actual_name, actual) in zip(
                baseline.named_parameters(), candidate.named_parameters()):
            assert name == actual_name
            torch.testing.assert_close(actual, ref, atol=4e-5, rtol=3e-4,
                                       msg=lambda message: f"{name}: {message}")
        if rank == 0:
            print(f"step {i + 1}: loss and parameters match", flush=True)
    ft._aggregated_tile_size = tile
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

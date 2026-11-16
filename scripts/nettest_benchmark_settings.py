"""Benchmark settings based on the upstream training recipe.

Batch-level lambda jitter is intentionally disabled: the sharded runner does
not yet implement it. These settings do not constitute an exact training run.
"""

from data_loader.config import DataloaderIOConfig, DataloaderSkipConfig
from model.config import LambdaConfig, LossParams, ModelConfig, NNUEConfig
from model.optimizers import OptimizerConfig


def make_nettest_settings(remap_table):
    config = NNUEConfig(
        features="Full_Threats+PP_3Wide+K32H3^",
        init_seed=1,
        model_config=ModelConfig(L1=1024, L2=32, L3=32, ls_remap_table=remap_table),
        optimizer_config=OptimizerConfig(
            optimizer_name="adamw", adamw_fused=True, lr=4e-4,
            factorized_weight_decay=4e-5, ft_weight_decay=0.02,
            virtual_ft_weight_decay=0.02, ft_bucket_lr_mult=0.25,
            threat_lr_mult=0.5, one_cycle_steps=5108000,
            one_cycle_warmup_pct=0.05, one_cycle_final_div=1000,
        ),
        loss_params=LossParams(
            pow_exp=2.4340402395048404, qp_asymmetry=0.22866886710086187,
            in_scaling=295.6539508488627, out_scaling=379.98724077106635,
            in_offset=285.2706341467852, out_offset=289.1258344152218,
            lambda_config=LambdaConfig(
                start_lambda=1.0, end_lambda=1.0, jitter_lambda_sample=0.0035,
                jitter_lambda_batch=0.0, jitter_decay_lambda_batch=0.999,
                lambda_cycle_jitter=True, lambda_schedule_steps=5052000,
                lambda_cycle_warmup_pct=0.25, lambda_cycle_delta=-0.3,
            ),
        ),
    )
    skip = DataloaderSkipConfig(
        random_fen_skipping=3, early_fen_skipping=18, soft_early_fen_skipping=32,
        pc_y0=-0.20, pc_y1=0.45, pc_y2=1.0, pc_y3=0.95, pc_y4=0.75,
        ply_x1=0.0, ply_y1=0.025, ply_x2=22.0, ply_y2=0.05,
        ply_x3=25.5, ply_y3=0.20, ply_x4=29.5, ply_y4=0.80,
        tree_sample_accept=0.12,
    )
    return config, skip, DataloaderIOConfig()

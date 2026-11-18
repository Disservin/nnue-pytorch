from pathlib import Path

from model.modules.stack_table import table_from_str
from scripts.nettest_benchmark_settings import make_nettest_settings


def test_nettest_benchmark_settings_are_explicit_about_batch_jitter():
    config, skip, io = make_nettest_settings("1:0")
    assert config.features == "Full_Threats+PP_3Wide+K32H3^"
    assert config.model_config.ls_remap_table == "1:0"
    assert (config.model_config.L1, config.model_config.L2, config.model_config.L3) == (1024, 32, 32)
    assert config.optimizer_config.optimizer_name == "adamw"
    assert config.optimizer_config.ft_bucket_lr_mult == 0.25
    assert config.optimizer_config.threat_lr_mult == 0.5
    assert config.optimizer_config.ft_weight_decay == 0.02
    assert config.loss_params.lambda_config.jitter_lambda_sample == 0.0035
    assert config.loss_params.lambda_config.jitter_lambda_batch == 0.007
    assert config.loss_params.lambda_config.jitter_decay_lambda_batch == 0.999
    assert (skip.random_fen_skipping, skip.early_fen_skipping, skip.soft_early_fen_skipping) == (3, 18, 32)
    assert skip.tree_sample_accept == 0.12
    assert io.shuffle_buffer_entries == 1048576


def test_saved_nettest_stack_table_has_consistent_fallbacks():
    path = Path(__file__).resolve().parents[1] / "docs" / "nettest_stack_remap.txt"
    mapping = table_from_str(path.read_text().strip(), 192)
    assert sum(i == fallback for i, fallback in enumerate(mapping)) == 92
    assert all(mapping[fallback] == fallback for fallback in mapping)

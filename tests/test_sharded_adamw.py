"""Exercise the AdamW sharded model and per-rank resume through real steps."""

import copy
import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_sharded_ft import _Feature, _local_batch

import model.model as model_module
import model.sharded_model as sharded_model_module
from model.config import LossParams, ModelConfig, NNUEConfig
from model.nnue import NNUE
from model.optimizers import OptimizerConfig
from trainer.sharded_adamw import ShardedAdamWStep


def _config():
    return NNUEConfig(
        features="Test",
        model_config=ModelConfig(
            L1=16, L2=4, L3=3, ls_keys="queen", ls_factors="queen", ls_remap_table="3:1"
        ),
        optimizer_config=OptimizerConfig(
            optimizer_name="adamw",
            adamw_fused=False,
            lr=0.0008,
            ft_weight_decay=0.03,
            virtual_ft_weight_decay=0.02,
            dense_weight_decay=0.01,
            factorized_weight_decay=0.04,
        ),
        loss_params=LossParams(w1=8, qp_asymmetry=0.25),
    )


def _batch(rank, n, step):
    values = list(_local_batch(rank, n))
    positions = torch.arange(n, dtype=torch.float32).reshape(-1, 1)
    values[4] = (positions + rank + step).remainder(3)
    values[5] = positions * 230 + rank * 570 - 1200 + step * 50
    values[6] = (torch.arange(n, dtype=torch.int64) + rank + step).remainder(4)
    return tuple(values)


def _worker(rank, port, batch_jitter, overlap=False):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank=rank, world_size=4, timeout=timedelta(seconds=60)
    )
    try:
        model_module.get_feature_cls = lambda _: [_Feature]
        sharded_model_module.get_feature_cls = lambda _: [_Feature]
        config = _config()
        # Overlap is NCCL-only by default; force it to test the Gloo path.
        options = {"overlap_replica_gradients": overlap}
        if batch_jitter:
            cfg = config.loss_params.lambda_config
            cfg.start_lambda = cfg.end_lambda = 0.5
            cfg.jitter_lambda_batch = 0.15
            cfg.jitter_decay_lambda_batch = 0.8
        torch.manual_seed(42)
        full = NNUE(config, max_epoch=3, num_batches_per_epoch=5)
        with torch.no_grad():
            full.model.input.bias.uniform_(0.25, 0.4)
            for name, param in full.model.layer_stacks.named_parameters():
                if "factorized" in name or "group_factor" in name:
                    param.uniform_(-0.05, 0.05)
        reference_optimizer = full.configure_optimizers()[0][0]

        # Seed ranks differently: the step must synchronize replicated parameters.
        torch.manual_seed(100 + rank)
        sharded = NNUE(config, max_epoch=3, num_batches_per_epoch=5, sharding=(rank, 4))
        sharded.model.input.load_unsharded(full.model.input)
        sharded.model.layer_stacks.l1.load_unsharded(full.model.layer_stacks.l1)
        sharded.model.layer_stacks.l2.load_state_dict(
            full.model.layer_stacks.l2.state_dict()
        )
        sharded.model.layer_stacks.output.load_state_dict(
            full.model.layer_stacks.output.state_dict()
        )
        if rank:
            with torch.no_grad():
                sharded.model.layer_stacks.l1.linear.bias.add_(rank)
                sharded.model.layer_stacks.l2.linear.weight.add_(rank)
                if batch_jitter:
                    sharded.lambda_scheduler.jitter_buffer.fill_(rank)
        runner = ShardedAdamWStep(sharded, gradient_clip_val=0.2, **options)
        reference_rng = runner.state_dict().get("batch_jitter_rng")
        snapshot = None
        for step in range(3):
            sizes = ([2, 0, 3, 1], [1, 2, 0, 4], [0, 1, 4, 1])[step]
            if overlap and step < 2:
                # Overlap applies only to equal local batches.
                sizes = [2] * 4
            batches = [_batch(i, n, step) for i, n in enumerate(sizes)]
            global_batch = tuple(
                torch.cat([part[i] for part in batches]) for i in range(7)
            )
            reference_optimizer.zero_grad(set_to_none=True)
            if batch_jitter:
                with torch.random.fork_rng(devices=[]):
                    torch.set_rng_state(reference_rng)
                    loss = full.compute_loss(global_batch, step)
                    reference_rng = torch.get_rng_state()
                torch.rand(rank + step + 1)
            else:
                loss = full.compute_loss(global_batch, step)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(full.model.parameters(), 0.2)
            reference_optimizer.step()
            got = runner.step(batches[rank])
            torch.testing.assert_close(got, loss.detach(), atol=3e-6, rtol=2e-5)
            if batch_jitter:
                jitter = sharded.lambda_scheduler.jitter_buffer
                all_jitter = [torch.empty_like(jitter) for _ in range(4)]
                dist.all_gather(all_jitter, jitter)
                for value in all_jitter:
                    torch.testing.assert_close(value, full.lambda_scheduler.jitter_buffer,
                                               atol=0, rtol=0)
            expected = dict(full.model.named_parameters())
            for name, param in sharded.model.named_parameters():
                reference = expected[name]
                if name.startswith("input.") or (
                    name.startswith("layer_stacks.l1.") and param.ndim == 2
                ):
                    reference = sharded.model.input.layout.select_columns(reference)
                torch.testing.assert_close(param, reference, atol=4e-6, rtol=3e-5)
            if step == 0:
                snapshot = copy.deepcopy(runner.state_dict())
            if step == 1:
                after_second = copy.deepcopy(runner.state_dict())
                resumed = NNUE(
                    config, max_epoch=3, num_batches_per_epoch=5, sharding=(rank, 4)
                )
                restarted = ShardedAdamWStep(resumed, gradient_clip_val=0.2, **options)
                restarted.load_state_dict(snapshot)
                assert restarted.global_step == snapshot["global_step"]
                assert resumed.max_steps == snapshot["max_steps"]
                assert restarted.schedulers[0].state_dict() == snapshot["schedulers"][0]
                resumed_loss = restarted.step(batches[rank])
                torch.testing.assert_close(resumed_loss, got, atol=2e-6, rtol=2e-5)
                resumed_state = restarted.state_dict()
                if batch_jitter:
                    assert torch.equal(resumed_state["batch_jitter_rng"], reference_rng)
                for name, value in after_second["nnue"].items():
                    if torch.is_tensor(value):
                        torch.testing.assert_close(resumed_state["nnue"][name], value)
                original_opt = after_second["nnue"]["optimizer_state_dict"]["state"]
                resumed_opt = resumed_state["nnue"]["optimizer_state_dict"]["state"]
                for param_id in original_opt:
                    for key, value in original_opt[param_id].items():
                        if torch.is_tensor(value):
                            torch.testing.assert_close(
                                resumed_opt[param_id][key], value
                            )
                bad = dict(snapshot, rank=(rank + 1) % 4)
                with pytest.raises(ValueError, match="checkpoint layout"):
                    restarted.load_state_dict(bad)
                bad_config = copy.deepcopy(snapshot)
                bad_config["config"]["loss_params"]["w1"] += 1
                with pytest.raises(ValueError, match="checkpoint layout"):
                    restarted.load_state_dict(bad_config)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable"
)
@pytest.mark.parametrize(
    ("batch_jitter", "overlap"),
    [(False, False), (True, False), (True, True)],
    ids=["no-jitter", "batch-jitter", "overlap"],
)
def test_four_rank_adamw_and_resume(batch_jitter, overlap):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, batch_jitter, overlap), nprocs=4, join=True)


def test_reject_non_adamw():
    config = _config()
    config.optimizer_config.optimizer_name = "rangerlite"
    with pytest.raises(ValueError, match="only AdamW"):
        NNUE(config, sharding=(0, 1))


def test_compilation_selection_on_cpu(monkeypatch):
    monkeypatch.setattr(sharded_model_module, "get_feature_cls", lambda _: [_Feature])
    calls = []
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: calls.append(kwargs) or fn)
    eager = NNUE(_config(), sharding=(0, 1))
    eager.model.layer_stacks.l1._compile_l1_merge = True
    assert not ShardedAdamWStep(eager).compile_model
    assert not eager.model.layer_stacks.l1._compile_l1_merge
    assert not calls
    requested = NNUE(_config(), sharding=(0, 1))
    monkeypatch.setattr(requested, "enable_compiled_loss", lambda: calls.append("loss"))
    assert ShardedAdamWStep(requested, compile_model=True).compile_model
    assert calls == [{"fullgraph": True}, {}, "loss"]


def test_real_features_allocate_only_shard_width():
    config = _config()
    config.features = "Full_Threats+PP_3Wide+K32H3^"
    nnue = NNUE(config, sharding=(2, 4))
    assert all(feature.weight.shape[1] == 4 for feature in nnue.model.input.features)
    assert nnue.model.input.bias.shape == (4,)
    assert nnue.model.layer_stacks.l1.linear.weight.shape[1] == 4
    nnue.model.clip_weights(include_input=True)


def test_stochastic_resume(monkeypatch):
    monkeypatch.setattr(sharded_model_module, "get_feature_cls", lambda _: [_Feature])
    config = _config()
    config.loss_params.lambda_config.jitter_lambda_sample = 0.2
    config.loss_params.lambda_config.start_lambda = 0.5
    config.loss_params.lambda_config.end_lambda = 0.5
    torch.manual_seed(902)
    nnue = NNUE(config, max_epoch=3, num_batches_per_epoch=5, sharding=(0, 1))
    runner = ShardedAdamWStep(nnue)
    batch = _batch(0, 3, 0)
    runner.step(batch)
    saved = copy.deepcopy(runner.state_dict())
    expected_loss = runner.step(batch)
    expected = copy.deepcopy(runner.state_dict())
    runner.load_state_dict(saved)
    torch.testing.assert_close(runner.step(batch), expected_loss)
    actual = runner.state_dict()
    for name, value in expected["nnue"].items():
        if torch.is_tensor(value):
            torch.testing.assert_close(actual["nnue"][name], value)

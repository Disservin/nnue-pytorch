import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from model.optimizers import OptimizerConfig, SafeOneCycleLR, WSDLR
from model.optimizers.lr_scheduler import setup_lr_scheduler, wsd_factor
from trainer.engine import SimpleTrainer, is_one_cycle


def _opt(lr=1.0):
    p = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.SGD([p], lr=lr)


def test_wsd_factor_phases():
    kw = dict(
        total_steps=1000,
        warmup_steps=50,
        decay_steps=200,
        shape="sqrt",
        start_div=25.0,
        final_div=1000.0,
    )
    assert wsd_factor(0, **kw) == pytest.approx(1 / 25)
    assert wsd_factor(50, **kw) == 1.0
    assert wsd_factor(799, **kw) == 1.0
    # 1-sqrt: a quarter into the decay leaves half of the decayable range
    f = wsd_factor(850, **kw)
    assert f == pytest.approx(1 / 1000 + (1 - 1 / 1000) * 0.5)
    assert wsd_factor(1000, **kw) == pytest.approx(1 / 1000)
    assert wsd_factor(5000, **kw) == pytest.approx(1 / 1000)
    # monotone non-increasing after warmup
    vals = [wsd_factor(s, **kw) for s in range(50, 1001)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))


@pytest.mark.parametrize("shape", ["sqrt", "linear", "cosine"])
def test_wsd_decay_endpoints(shape):
    kw = dict(
        total_steps=100,
        warmup_steps=0,
        decay_steps=40,
        shape=shape,
        start_div=25.0,
        final_div=100.0,
    )
    assert wsd_factor(60, **kw) == pytest.approx(1.0)
    assert wsd_factor(100, **kw) == pytest.approx(0.01)


def test_setup_wsd_is_stepped_per_batch():
    cfg = OptimizerConfig(
        optimizer_name="adamw",
        lr_schedule="wsd",
        wsd_total_steps=1000,
        wsd_warmup_pct=0.05,
        wsd_decay_pct=0.2,
    )
    sch = setup_lr_scheduler(_opt(), cfg)
    assert isinstance(sch["scheduler"], WSDLR)
    assert is_one_cycle(sch["scheduler"])


def test_setup_wsd_requires_total_steps():
    cfg = OptimizerConfig(optimizer_name="adamw", lr_schedule="wsd")
    with pytest.raises(ValueError):
        setup_lr_scheduler(_opt(), cfg)


def test_wsd_resume_uses_new_shape_and_old_position():
    opt = _opt(lr=1.0)
    sch = WSDLR(opt, total_steps=1000, warmup_steps=0, decay_steps=200)
    for _ in range(500):
        sch.step()
    state = sch.state_dict()

    # Resume with a longer schedule: still in the stable phase at step 500.
    opt2 = _opt(lr=1.0)
    sch2 = WSDLR(opt2, total_steps=4000, warmup_steps=0, decay_steps=800)
    sch2.load_state_dict(state)
    assert sch2.last_epoch == 500
    assert opt2.param_groups[0]["lr"] == pytest.approx(1.0)

    # Resume with a schedule whose decay already started.
    opt3 = _opt(lr=1.0)
    sch3 = WSDLR(opt3, total_steps=600, warmup_steps=0, decay_steps=200)
    sch3.load_state_dict(state)
    expected = 1 / 25000 + (1 - 1 / 25000) * (1 - math.sqrt(0.5))
    assert opt3.param_groups[0]["lr"] == pytest.approx(expected)


def test_wsd_accepts_onecycle_state():
    opt = _opt(lr=1.0)
    oc = SafeOneCycleLR(opt, max_lr=[1.0], total_steps=100, cycle_momentum=False)
    for _ in range(30):
        oc.step()
    opt2 = _opt(lr=1.0)
    sch = WSDLR(opt2, total_steps=100, warmup_steps=0, decay_steps=50)
    sch.load_state_dict(oc.state_dict())
    assert sch.last_epoch == 30
    assert opt2.param_groups[0]["lr"] == pytest.approx(1.0)


def _trainer(max_epochs=10):
    model = torch.nn.Linear(2, 2)
    return SimpleTrainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        schedulers=[],
        max_epochs=max_epochs,
        check_val_every_n_epoch=1,
        gradient_clip_val=None,
        log_every_n_steps=1,
        default_root_dir=None,
        callbacks=[],
        logger=None,
        device="cpu",
        rank=0,
        world_size=1,
        local_rank=0,
    )


@pytest.mark.parametrize("flag,expected_epoch", [(True, 5), (False, 4), (None, 5)])
def test_resume_continues_after_finished_epoch(tmp_path, flag, expected_epoch):
    t = _trainer()
    t.current_epoch = 4
    t.global_step = 123
    t.epoch_finished = bool(flag)
    path = str(tmp_path / "last.ckpt")
    t.save_checkpoint(path)
    if flag is None:
        # legacy checkpoint without the flag
        ckpt = torch.load(path, weights_only=False)
        del ckpt["epoch_finished"]
        torch.save(ckpt, path)

    t2 = _trainer()
    t2.load_checkpoint(path)
    assert t2.current_epoch == expected_epoch
    assert t2.global_step == 123


def test_fit_is_noop_when_all_epochs_done(tmp_path):
    t = _trainer(max_epochs=5)
    t.current_epoch = 4
    t.epoch_finished = True
    path = str(tmp_path / "last.ckpt")
    t.save_checkpoint(path)

    t2 = _trainer(max_epochs=5)
    t2.load_checkpoint(path)
    t2.fit(train_loader=[])  # must not iterate or call any hook
    assert t2.current_epoch == 5

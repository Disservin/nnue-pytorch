import math
from dataclasses import dataclass
from typing import Any, Literal

import torch


@dataclass(kw_only=True, frozen=False)
class LRSchedulerConfig:
    gamma: float = 0.992
    """Multiplicative factor applied to the learning rate after every epoch."""

    lr_schedule: Literal["onecycle", "wsd"] = "onecycle"
    """LR schedule. 'onecycle' uses --one-cycle-steps (StepLR with --gamma when that is <= 0). 'wsd' is warmup-stable-decay, configured by the --wsd-* options."""

    one_cycle_steps: int = 0
    """Number of steps for the One Cycle LR scheduler. If set to a positive value, One Cycle LR scheduler will be used. If set to 0 or a negative value, StepLR with step_size=1 will be used."""

    one_cycle_warmup_pct: float = 0.2
    """Fraction of the cycle to spend increasing the learning rate in the One Cycle LR scheduler."""

    one_cycle_start_div: float = 25
    """Initial lr div factor when using One Cycle LR scheduler."""

    one_cycle_final_div: float = 50
    """Final lr div factor when using One Cycle LR scheduler."""

    wsd_total_steps: int = 0
    """Total optimizer steps of the WSD schedule; the decay ends here. The schedule depends only on the global step and is rebuilt from the current options on resume, so a stable-phase checkpoint can be resumed with a different --wsd-total-steps to branch off a cooldown."""

    wsd_warmup_pct: float = 0.05
    """Fraction of --wsd-total-steps spent warming up (cosine ramp from lr / --wsd-start-div)."""

    wsd_decay_pct: float = 0.2
    """Fraction of --wsd-total-steps spent in the final decay."""

    wsd_decay_shape: Literal["sqrt", "linear", "cosine"] = "sqrt"
    """Decay shape. 'sqrt' is the 1-sqrt cooldown (Hagele et al. 2024)."""

    wsd_start_div: float = 25.0
    """The warmup starts at lr / wsd_start_div."""

    wsd_final_div: float = 25000.0
    """The decay ends at lr / wsd_final_div. The default equals the final LR of OneCycle with start_div=25 and final_div=1e3."""


class SafeOneCycleLR(torch.optim.lr_scheduler.OneCycleLR):
    def step(self, epoch=None):
        if self.last_epoch < self.total_steps - 1:
            super().step(epoch)


def wsd_factor(
    step: int,
    total_steps: int,
    warmup_steps: int,
    decay_steps: int,
    shape: str,
    start_div: float,
    final_div: float,
) -> float:
    """LR multiplier of the warmup-stable-decay schedule at a global step."""
    final = 1.0 / final_div
    if step < warmup_steps:
        start = 1.0 / start_div
        t = step / warmup_steps
        return start + (1.0 - start) * 0.5 * (1.0 - math.cos(math.pi * t))
    decay_start = total_steps - decay_steps
    if step < decay_start:
        return 1.0
    if step >= total_steps:
        return final
    t = (step - decay_start) / decay_steps
    if shape == "sqrt":
        w = 1.0 - math.sqrt(t)
    elif shape == "linear":
        w = 1.0 - t
    elif shape == "cosine":
        w = 0.5 * (1.0 + math.cos(math.pi * t))
    else:
        raise ValueError(f"Unknown WSD decay shape: {shape!r}")
    return final + (1.0 - final) * w


class WSDLR(torch.optim.lr_scheduler.LRScheduler):
    """Warmup-stable-decay schedule, stepped once per optimizer step.

    Only the step position is checkpointed; the shape always comes from the
    options of the current run. A long stable phase can therefore be trained
    once and several cooldowns of different lengths branched off its
    checkpoints, and a OneCycle checkpoint can be resumed under WSD.
    """

    per_step = True

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int,
        decay_steps: int,
        shape: str = "sqrt",
        start_div: float = 25.0,
        final_div: float = 25000.0,
        last_epoch: int = -1,
    ):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.shape = shape
        self.start_div = start_div
        self.final_div = final_div
        super().__init__(optimizer, last_epoch)

    def factor(self, step: int) -> float:
        return wsd_factor(
            step,
            self.total_steps,
            self.warmup_steps,
            self.decay_steps,
            self.shape,
            self.start_div,
            self.final_div,
        )

    def get_lr(self):
        f = self.factor(max(self.last_epoch, 0))
        return [base_lr * f for base_lr in self.base_lrs]

    def state_dict(self):
        return {"last_epoch": self.last_epoch, "_step_count": self._step_count}

    def load_state_dict(self, state_dict):
        # Take only the position from any step-based scheduler state (this
        # includes OneCycleLR); everything else follows the current options.
        self.last_epoch = int(state_dict["last_epoch"])
        self._step_count = int(state_dict.get("_step_count", self.last_epoch + 1))
        lrs = self.get_lr()
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr
        self._last_lr = lrs


def _make_wsd(optimizer, config) -> WSDLR:
    total = config.wsd_total_steps
    if total <= 0:
        raise ValueError("--lr-schedule=wsd requires --wsd-total-steps > 0.")
    warmup_pct, decay_pct = config.wsd_warmup_pct, config.wsd_decay_pct
    if not (0.0 <= warmup_pct <= 1.0 and 0.0 <= decay_pct <= 1.0):
        raise ValueError("--wsd-warmup-pct and --wsd-decay-pct must be in [0, 1].")
    if warmup_pct + decay_pct > 1.0:
        raise ValueError("--wsd-warmup-pct + --wsd-decay-pct must be <= 1.")
    if config.wsd_start_div <= 0.0 or config.wsd_final_div <= 0.0:
        raise ValueError("--wsd-start-div and --wsd-final-div must be positive.")
    return WSDLR(
        optimizer,
        total_steps=total,
        warmup_steps=int(round(config.wsd_warmup_pct * total)),
        decay_steps=int(round(config.wsd_decay_pct * total)),
        shape=config.wsd_decay_shape,
        start_div=config.wsd_start_div,
        final_div=config.wsd_final_div,
    )


def setup_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: LRSchedulerConfig,
) -> torch.optim.lr_scheduler.StepLR | dict[str, Any]:
    if config.lr_schedule == "wsd":
        return {"scheduler": _make_wsd(optimizer, config), "interval": "step"}

    if config.one_cycle_steps <= 0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=config.gamma
        )
    else:
        LRs = [group["lr"] for group in optimizer.param_groups]
        one_cycle_scheduler = SafeOneCycleLR(
            optimizer,
            max_lr=LRs,
            total_steps=config.one_cycle_steps,
            div_factor=config.one_cycle_start_div,
            final_div_factor=config.one_cycle_final_div,
            pct_start=config.one_cycle_warmup_pct,
            cycle_momentum=False,
        )
        scheduler = {"scheduler": one_cycle_scheduler, "interval": "step"}

    return scheduler

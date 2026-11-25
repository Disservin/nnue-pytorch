"""SimpleTrainer variant for FT/L1-sharded data-parallel training.

The sharded runner owns the forward/backward/optimizer step. This trainer
adds the pieces the generic loop cannot provide for that layout: validation
through the sharded forward, and checkpoints that can be consumed by the
regular full-model tooling while still restoring per-rank optimizer state.
"""

from __future__ import annotations

import os
import typing

import torch
import torch.distributed as dist

from model.model import NNUEModel

from .checkpoint_writer import _atomic_save
from .engine import SimpleTrainer, _unwrap_module

if typing.TYPE_CHECKING:  # pragma: no cover
    from .sharded_adamw import ShardedAdamWStep


CHECKPOINT_FORMAT = "nnue-sharded-v1"


class ShardedTrainer(SimpleTrainer):
    """Train a ``ShardedNNUEModel`` with a ``ShardedAdamWStep`` runner.

    ``save_checkpoint`` writes two artifacts for every path:

    * ``path`` - a regular checkpoint whose ``state_dict`` holds the complete,
      gathered model (usable by the normal conversion and resume tooling).
    * ``path.rank<N>`` - rank N's shard weights, AdamW state, scheduler and
      RNG state. Resume requires one file per rank and the same world size;
      without them the rank's optimizer starts fresh.
    """

    wrap_in_ddp = False

    def __init__(self, *, step_runner: ShardedAdamWStep, **kwargs):
        self.step_runner = step_runner
        super().__init__(
            optimizer=step_runner.optimizer, schedulers=step_runner.schedulers, **kwargs
        )

    # The runner owns zero_grad, backward, clipping, the optimizer step and
    # the per-step schedulers.
    def _optimization_step(self, batch):
        loss = self.step_runner.step(batch).detach()
        self.model.loss_metrics["train_loss_epoch"].update(loss)
        self.global_step = self.step_runner.global_step
        return {"loss": loss, "train_loss": loss}

    def _validation_step(self, batch):
        val_loss = self.step_runner.val_step(batch)
        self.model.loss_metrics["val_loss_epoch"].update(val_loss)
        return {"val_loss": val_loss}

    # -- checkpoints --------------------------------------------------
    def _full_model_for_save(self):
        model = _unwrap_module(self.model).model
        if self.rank != 0:
            return None
        # NNUEModel's constructor initializes weights; keep the training RNG
        # streams untouched.
        with torch.random.fork_rng(devices=[]):
            full = NNUEModel(model.feature_name, model.model_config)
        return full.to(self.device)

    def _shared_path(self, path: str) -> str:
        """Return *path* under rank 0's directory.

        Loggers pick a new version directory per process, so ranks disagree on
        the checkpoint directory. Rank states must still land next to the
        merged checkpoint that rank 0 writes.
        """
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return path
        payload = [os.path.dirname(path) if self.rank == 0 else None]
        dist.broadcast_object_list(payload, src=0)
        return os.path.join(payload[0], os.path.basename(path))

    def save_checkpoint(self, path: str):
        unwrapped = _unwrap_module(self.model)
        full = self._full_model_for_save()
        unwrapped.model.collect_unsharded_into(full, group=None)

        rank_state = self.step_runner.state_dict()
        rank_path = f"{self._shared_path(path)}.rank{self.rank}"
        _atomic_save(rank_state, rank_path)

        if self.rank == 0:
            state = {
                f"model.{name}": value.detach().cpu()
                for name, value in full.state_dict().items()
            }
            state["lambda_scheduler_state_dict"] = (
                unwrapped.lambda_scheduler.state_dict()
            )
            state["jitter_buffer_value"] = (
                unwrapped.lambda_scheduler.jitter_buffer.detach().cpu()
            )
            checkpoint = {
                "format": CHECKPOINT_FORMAT,
                "epoch": self.current_epoch,
                "global_step": self.global_step,
                "state_dict": state,
                "lr_schedulers": [sch.state_dict() for sch in self._schedulers],
            }
            self._call_callbacks("on_save_checkpoint", self, checkpoint)
            if self.checkpoint_writer is not None:
                self.checkpoint_writer.submit(checkpoint, path)
            else:
                _atomic_save(checkpoint, path)
            del full

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(
                f"{path} is not a sharded-training checkpoint; resume sharded "
                "runs from their own last.ckpt"
            )

        unwrapped = _unwrap_module(self.model)
        model = unwrapped.model
        with torch.random.fork_rng(devices=[]):
            full = NNUEModel(model.feature_name, model.model_config)
        state = dict(checkpoint["state_dict"])
        lambda_state = state.pop("lambda_scheduler_state_dict", None)
        jitter_value = state.pop("jitter_buffer_value", None)
        full.load_state_dict(
            {name.removeprefix("model."): value for name, value in state.items()}
        )
        model.load_unsharded(full)
        if lambda_state is not None:
            unwrapped.lambda_scheduler.load_state_dict(lambda_state)
        if jitter_value is not None:
            unwrapped.lambda_scheduler.jitter_buffer.copy_(
                jitter_value.to(
                    device=unwrapped.lambda_scheduler.jitter_buffer.device,
                    dtype=unwrapped.lambda_scheduler.jitter_buffer.dtype,
                )
            )
        del full

        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)

        rank_path = f"{path}.rank{self.rank}"
        if os.path.exists(rank_path):
            rank_state = torch.load(
                rank_path, map_location=self.device, weights_only=False
            )
            self.step_runner.load_state_dict(rank_state)
        elif dist.is_initialized() and dist.get_world_size() > 1:
            raise RuntimeError(
                f"missing per-rank sharded state {rank_path}; sharded resume "
                "requires the same world size and every rank's state file"
            )
        else:
            self.step_runner.global_step = self.global_step

        scheduler_states = checkpoint.get("lr_schedulers", [])
        for sch, value in zip(self._schedulers, scheduler_states):
            sch.load_state_dict(value)
        self._call_callbacks("on_load_checkpoint", self, checkpoint)
        return checkpoint

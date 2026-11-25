"""AdamW step for the FT/L1-sharded NNUE model.

The runner owns zero_grad, forward, backward, gradient synchronization,
clipping, the optimizer step and the per-step schedulers. Its state is per
rank and requires the same process-group layout on resume.
"""

from dataclasses import asdict

import torch
import torch.distributed as dist

from model.sharded_backward import backward_weighted_loss, clip_sharded_grad_norm_
from model.sharded_model import ShardedNNUEModel
from trainer.engine import is_one_cycle


class ShardedAdamWStep:
    """Run AdamW on FT/L1 shards and a data-parallel tail.

    Every rank must call ``step`` and ``val_step`` in the same order, with
    equal-size local batches. On CUDA the local FT merge, post-L1 tail and
    loss are compiled unless ``compile_model=False``. The replica gradient
    reduction overlaps the local FT backward.
    """

    def __init__(self, nnue, *, group=None, gradient_clip_val=2.0,
                 compile_model: bool | None = None):
        if not isinstance(nnue.model, ShardedNNUEModel):
            raise TypeError("ShardedAdamWStep requires a sharded NNUE model")
        if nnue.config.optimizer_config.optimizer_name.lower() != "adamw":
            raise ValueError("sharded training supports only AdamW")
        if nnue.optimizer_wrapper is not None:
            raise ValueError("ShardedAdamWStep must create the NNUE optimizer")
        self.nnue, self.group = nnue, group
        self.gradient_clip_val = gradient_clip_val
        self.world = (
            dist.get_world_size(group)
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        self.rank = dist.get_rank(group) if self.world > 1 else 0
        layout = nnue.model.input.layout
        if (layout.rank, layout.world_size) != (self.rank, self.world):
            raise ValueError("shard layout does not match the process group")
        self.sharded = nnue.model.sharded_parameters()
        self.replicated = nnue.model.replicated_parameters()
        device = next(nnue.parameters()).device
        self.compile_model = device.type == "cuda" if compile_model is None else compile_model
        self._batch_jitter_generator = None
        # Replicas may have been created with different process seeds. Broadcast
        # them before creating optimizer state; never broadcast shard weights.
        if self.world > 1:
            src = dist.get_global_rank(group, 0) if group is not None else 0
            with torch.no_grad():
                for p in self.replicated:
                    dist.broadcast(p, src=src, group=group)
                for buffer in nnue.model.layer_stacks.buffers():
                    dist.broadcast(buffer, src=src, group=group)
                if nnue.config.loss_params.lambda_config.jitter_lambda_batch:
                    self._init_batch_jitter(src, device)
        nnue.model.layer_stacks.l1._compile_l1_merge = self.compile_model
        if self.compile_model:
            # Compile only the local pieces. The distributed forward itself
            # includes collectives and remains eager.
            nnue.model.input._merge_weight_and_bias_torch = torch.compile(
                nnue.model.input._merge_weight_and_bias_torch, fullgraph=True
            )
            nnue.model.layer_stacks.forward_from_l1 = torch.compile(
                nnue.model.layer_stacks.forward_from_l1
            )
            nnue.enable_compiled_loss()
        optimizers, schedulers = nnue.configure_optimizers()
        if len(optimizers) != 1 or not isinstance(optimizers[0], torch.optim.AdamW):
            raise ValueError("expected exactly one AdamW optimizer")
        self.optimizer = optimizers[0]
        self.schedulers = [
            s.get("scheduler", s) if isinstance(s, dict) else s for s in schedulers
        ]
        self.global_step = 0

    def _init_batch_jitter(self, src, device):
        # Each rank draws one scalar per global step from its own copy of this
        # stream. Local sample jitter and unrelated RNG calls cannot advance it.
        if self.rank == 0:
            seed = torch.empty((), dtype=torch.int64, device="cpu").random_().to(device)
        else:
            seed = torch.empty((), dtype=torch.int64, device=device)
        dist.broadcast(seed, src=src, group=self.group)
        dist.broadcast(self.nnue.lambda_scheduler.jitter_buffer, src=src, group=self.group)
        self._batch_jitter_generator = torch.Generator(device=device).manual_seed(seed.item())

    def _fake_quantize(self):
        enabled = self.global_step >= self.nnue.config.fake_quant_warmup_steps
        return (
            self.nnue.config.use_fake_act_quantization and enabled,
            self.nnue.config.use_fake_weight_quantization and enabled,
        )

    def step(self, batch):
        """Run one local-batch forward/backward and return the global loss.

        The batch and the model must already be on the collective device.
        """
        if not self.nnue.training:
            raise RuntimeError("sharded AdamW steps require training mode")
        self.optimizer.zero_grad(set_to_none=True)
        global_batch = self.nnue.model.gather_batch(batch, self.group)
        prediction, ft_boundary = self.nnue.model.forward_global_batch(
            global_batch, *self._fake_quantize(), self.group
        )
        numerator, weight = self.nnue.compute_loss_components_with_scorenet(
            prediction, batch, self.global_step,
            batch_jitter_generator=self._batch_jitter_generator,
        )
        contribution = backward_weighted_loss(
            numerator, weight, self.replicated, ft_boundary, self.group
        )
        if self.gradient_clip_val is not None and self.gradient_clip_val > 0:
            clip_sharded_grad_norm_(
                self.sharded, self.replicated, self.gradient_clip_val, self.group
            )
        self.optimizer.step()
        for scheduler in self.schedulers:
            if is_one_cycle(scheduler):
                scheduler.step()
        self.global_step += 1
        loss = contribution.clone()
        if self.world > 1:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=self.group)
        return loss

    @torch.no_grad()
    def val_step(self, batch):
        """Return the global validation loss for one local batch.

        Every rank gathers the local feature batches, runs the sharded
        forward in evaluation mode and reduces the weighted loss components.
        The batch must already be on the collective device.
        """
        if self.nnue.training:
            raise RuntimeError("sharded validation steps require evaluation mode")
        global_batch = self.nnue.model.gather_batch(batch, self.group)
        prediction, _ = self.nnue.model.forward_global_batch(
            global_batch, *self._fake_quantize(), self.group
        )
        numerator, weight = self.nnue.compute_loss_components_with_scorenet(
            prediction, batch, self.global_step
        )
        if self.world > 1:
            dist.all_reduce(numerator, op=dist.ReduceOp.SUM, group=self.group)
            dist.all_reduce(weight, op=dist.ReduceOp.SUM, group=self.group)
        return numerator / weight

    def _group_parameter_names(self):
        names = {id(p): name for name, p in self.nnue.named_parameters()}
        return [
            [names[id(p)] for p in group["params"]]
            for group in self.optimizer.param_groups
        ]

    def state_dict(self):
        """Return rank state with PyTorch's normal shallow tensor references.

        Serialize or deep-copy it before further optimizer steps. Each rank must
        save its own checkpoint.
        """
        device = next(self.nnue.parameters()).device
        state = {
            "format": "sharded-adamw-v1",
            "rank": self.rank,
            "world_size": self.world,
            "global_step": self.global_step,
            "parameter_groups": self._group_parameter_names(),
            "config": asdict(self.nnue.config),
            "max_steps": self.nnue.max_steps,
            "gradient_clip_val": self.gradient_clip_val,
            "device_type": device.type,
            "nnue": self.nnue.state_dict(),
            "schedulers": [scheduler.state_dict() for scheduler in self.schedulers],
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device)
            if device.type == "cuda"
            else None,
        }
        if self._batch_jitter_generator is not None:
            state["batch_jitter_rng"] = self._batch_jitter_generator.get_state()
        return state

    def load_state_dict(self, state):
        """Restore a checkpoint produced by the same rank and world size."""
        batch_jitter_rng = state.get("batch_jitter_rng")
        if (
            state.get("format") != "sharded-adamw-v1"
            or state.get("rank") != self.rank
            or state.get("world_size") != self.world
            or state.get("parameter_groups") != self._group_parameter_names()
            or state.get("config") != asdict(self.nnue.config)
            or state.get("gradient_clip_val") != self.gradient_clip_val
            or state.get("device_type") != next(self.nnue.parameters()).device.type
            or len(state.get("schedulers", ())) != len(self.schedulers)
            or (batch_jitter_rng is None) != (self._batch_jitter_generator is None)
        ):
            raise ValueError("incompatible sharded AdamW checkpoint layout")
        self.nnue.load_state_dict(state["nnue"])
        for scheduler, value in zip(self.schedulers, state["schedulers"]):
            scheduler.load_state_dict(value)
        self.global_step = state["global_step"]
        self.nnue.max_steps = state["max_steps"]
        if self._batch_jitter_generator is not None:
            self._batch_jitter_generator.set_state(batch_jitter_rng.cpu())
        torch.set_rng_state(state["cpu_rng"].cpu())
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state(
                state["cuda_rng"].cpu(), next(self.nnue.parameters()).device
            )

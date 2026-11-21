"""Opt-in AdamW step for the FT/L1-sharded NNUE model.

This is a standalone training building block, not SimpleTrainer's DDP mode.
Checkpoints are per rank and require the same process-group layout on resume.
"""

from dataclasses import asdict
from itertools import pairwise

import torch
import torch.distributed as dist

from model.sharded_backward import backward_weighted_loss, clip_sharded_grad_norm_
from model.sharded_model import ShardedNNUEModel
from trainer.engine import is_one_cycle


class ShardedAdamWStep:
    """Run AdamW on FT/L1 shards and a data-parallel tail.

    ``bucketed_bias`` opts into a guarded CUDA bias-gradient reduction; it
    falls back to eager for unsupported inputs and remains a caller choice
    when restoring a checkpoint.
    """
    def __init__(self, nnue, *, group=None, gradient_clip_val=2.0,
                 compile_model: bool | None = None, bucketed_bias: bool = False,
                 uniform_loss: bool = False, overlap_replica_gradients: bool = False,
                 direct_packed_ft: bool = False):
        if not isinstance(nnue.model, ShardedNNUEModel):
            raise TypeError(
                "ShardedAdamWStep requires a directly constructed sharded NNUE"
            )
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
        if {id(p) for p in self.sharded + self.replicated} != {
            id(p) for p in nnue.model.parameters()
        }:
            raise ValueError("sharded and replicated parameters do not cover the model")
        if len({id(p) for p in self.sharded + self.replicated}) != len(
            self.sharded
        ) + len(self.replicated):
            raise ValueError("sharded and replicated parameters must be disjoint")
        device = next(nnue.parameters()).device
        if compile_model is not None and not isinstance(compile_model, bool):
            raise TypeError("compile_model must be a bool or None")
        for name, value in (
            ("bucketed_bias", bucketed_bias), ("uniform_loss", uniform_loss),
            ("overlap_replica_gradients", overlap_replica_gradients),
            ("direct_packed_ft", direct_packed_ft),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool")
        self.compile_model = device.type == "cuda" if compile_model is None else compile_model
        self.bucketed_bias = bucketed_bias
        self.uniform_loss = uniform_loss
        self.overlap_replica_gradients = overlap_replica_gradients
        self.direct_packed_ft = direct_packed_ft
        self.last_experiments = {}
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
        nnue.model._compile_layer_clipping = self.compile_model
        nnue.model._direct_packed_ft = self.direct_packed_ft
        nnue.model.layer_stacks.l1._compile_l1_merge = self.compile_model
        nnue.model.layer_stacks.l1._bucketed_bias = self.bucketed_bias
        if self.compile_model:
            # Compile only the tested, local pieces. The distributed forward
            # itself includes collectives and remains eager.
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

    def step(self, batch, *, global_batch=None, after_forward=None):
        """Run one local-batch forward/backward and return the global loss.

        Batches and the model must already be on the collective device. Epoch
        schedulers and weight-clipping callbacks remain the caller's responsibility.
        ``global_batch`` and ``after_forward`` support experimental lookahead.
        All ranks must use matching batches and enqueue any callback collectives
        in the same order; the callback must not change model weights.
        """
        if not self.nnue.training:
            raise RuntimeError("sharded AdamW steps require training mode")
        if (
            self.world > 1
            and self._batch_jitter_generator is None
            and self.nnue.config.loss_params.lambda_config.jitter_lambda_batch
        ):
            raise ValueError(
                "enable batch lambda jitter before constructing the sharded runner"
            )
        self.optimizer.zero_grad(set_to_none=True)
        enabled = self.global_step >= self.nnue.config.fake_quant_warmup_steps
        loss_params = self.nnue.config.loss_params
        uniform = self.uniform_loss and loss_params.w1 == 0 and 0 <= loss_params.w2 < float("inf")
        overlap = (
            self.overlap_replica_gradients and self.world > 1 and bool(self.replicated)
            and all(p.requires_grad and p.dtype == torch.float32 for p in self.replicated)
            and any(p.requires_grad for p in self.nnue.model.input.parameters())
        )
        if global_batch is None and (uniform or overlap or self.direct_packed_ft):
            global_batch = self.nnue.model.gather_batch(batch, self.group)
        if overlap:
            offsets = global_batch.offsets
            local_size = offsets[1] - offsets[0]
            overlap = local_size > 0 and all(
                end - start == local_size for start, end in pairwise(offsets)
            )
        self.last_experiments = {
            "uniform_loss": uniform, "overlap_replica_gradients": overlap,
            "direct_packed_ft": global_batch is not None
            and global_batch.packed_payload is not None,
        }
        forward = (self.nnue.model.forward_batch if global_batch is None
                   else self.nnue.model.forward_global_batch)
        result = forward(
            batch if global_batch is None else global_batch,
            self.nnue.config.use_fake_act_quantization and enabled,
            self.nnue.config.use_fake_weight_quantization and enabled,
            self.group,
            **({"split_ft_backward": True} if overlap else {}),
        )
        prediction, ft_backward = result if overlap else (result, None)
        if after_forward is not None:
            after_forward()
        numerator, weight = self.nnue.compute_loss_components_with_scorenet(
            prediction, batch, self.global_step,
            batch_jitter_generator=self._batch_jitter_generator,
            uniform_weights=uniform,
        )
        contribution = backward_weighted_loss(
            numerator, weight, self.replicated, self.group,
            global_count=global_batch.offsets[-1] if uniform else None,
            ft_backward=ft_backward,
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

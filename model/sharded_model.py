"""Opt-in FT/L1-sharded NNUE model built without full FT parameters."""

import torch
import torch.distributed as dist
from torch import nn

from data_loader.distributed_features import gather_packed_features

from .model import NNUEModel
from .modules import LayerStacks, get_feature_cls
from .modules.feature_transformer import ShardedFeatureTransformer
from .modules.feature_transformer.fused_ft_functions import _HAS_CUPY_KERNELS
from .modules.layer_stacks import stack_count, stack_keys
from .modules.sharded_l1 import ShardedL1
from .quantize import QuantizationManager


class ShardedNNUEModel(NNUEModel):
    """Own one FT/L1 weight shard and the replicated data-parallel tail."""

    def __init__(self, feature_name, config, *, rank, world_size):
        nn.Module.__init__(self)
        self.model_config = config
        self.L1, self.L2, self.L3 = config.L1, config.L2, config.L3
        self.quantize_config = config.quantize_config
        self.quantization = QuantizationManager(config.quantize_config)
        self.ls_keys = config.ls_keys
        self.input = ShardedFeatureTransformer(
            get_feature_cls(feature_name),
            self.L1,
            self.quantization,
            rank=rank,
            world_size=world_size,
        )
        self.feature_name = self.input.FEATURE_NAME
        self.input_feature_name = self.input.INPUT_FEATURE_NAME
        self.feature_hash = self.input.HASH
        self.layer_stacks = LayerStacks(
            stack_count(stack_keys(config.ls_keys)),
            config,
            self.quantization,
            l1_factory=lambda width, out, count, quant: ShardedL1(
                width, out, count, quant, rank=rank, world_size=world_size
            ),
        )
        self.num_ls_buckets = self.layer_stacks.count
        self.weight_clipping = self.quantization.generate_weight_clipping_config(self)

    def sharded_parameters(self):
        return list(self.input.parameters()) + self.layer_stacks.l1.sharded_parameters()

    def replicated_parameters(self):
        return (
            self.layer_stacks.l1.replicated_bias_parameters()
            + list(self.layer_stacks.l2.parameters())
            + list(self.layer_stacks.output.parameters())
        )


    @torch.no_grad()
    def load_unsharded(self, full: NNUEModel) -> None:
        """Copy full-model weights into this rank's FT/L1 shard.

        The replicated tail is loaded from *full* unchanged. All ranks must
        call this with compatible models; no collectives are performed.
        """
        self.input.load_unsharded(full.input)
        self.layer_stacks.l1.load_unsharded(full.layer_stacks.l1)
        self.layer_stacks.l2.load_state_dict(full.layer_stacks.l2.state_dict())
        self.layer_stacks.output.load_state_dict(
            full.layer_stacks.output.state_dict()
        )

    @torch.no_grad()
    def collect_unsharded_into(self, full: NNUEModel | None, group=None) -> None:
        """Gather every shard into *full* on rank 0; other ranks pass None.

        All ranks must call this. Sharded FT and L1 weight columns are
        concatenated in rank order and scattered back into the paired-column
        layout of the full model. Replicated parameters, biases and buffers
        are copied from rank 0 after their equality has been established by
        the sharded AdamW bootstrap broadcast or by construction. NCCL has no
        gather, so every rank receives the parts; only rank 0 assembles them.
        """
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("collect_unsharded_into requires a process group")
        world = dist.get_world_size(group)
        rank = dist.get_rank(group)
        layout = self.input.layout
        if world != layout.world_size or rank != layout.rank:
            raise ValueError("shard layout does not match the process group")
        if (full is not None) != (rank == 0):
            raise ValueError("only rank 0 must pass the full model")

        sharded_ids = {id(p) for p in self.sharded_parameters()}
        full_params = dict(full.named_parameters()) if rank == 0 else None
        for name, src in self.named_parameters():
            if id(src) in sharded_ids:
                parts = [torch.empty_like(src) for _ in range(world)]
                dist.all_gather(parts, src.detach().contiguous(), group=group)
                if rank == 0:
                    dst = full_params[name]
                    width = layout.pair_width
                    half = layout.full_width // 2
                    for i, part in enumerate(parts):
                        dst[..., i * width : (i + 1) * width].copy_(
                            part[..., :width]
                        )
                        dst[..., half + i * width : half + (i + 1) * width].copy_(
                            part[..., width:]
                        )
            elif rank == 0:
                full_params[name].copy_(src)
        if rank == 0:
            full_buffers = dict(full.named_buffers())
            for name, src in self.named_buffers():
                if name in full_buffers:
                    full_buffers[name].copy_(src)

    def forward_batch(
        self, batch, fake_quantize_acts=True, fake_quantize_weights=True, group=None
    ):
        global_batch = self.gather_batch(batch, group)
        return self.forward_global_batch(global_batch, fake_quantize_acts, fake_quantize_weights, group)

    def gather_batch(self, batch, group=None):
        if self.can_use_direct_packed_ft():
            return gather_packed_features(batch, group, direct_packed_ft=True)
        return gather_packed_features(batch, group)

    def can_use_direct_packed_ft(self):
        device = self.input.bias.device
        if (
            not getattr(self, "_direct_packed_ft", False) or not _HAS_CUPY_KERNELS
            or device.type != "cuda" or torch.version.hip is not None
            or torch.is_autocast_enabled() or torch.are_deterministic_algorithms_enabled()
            or not all(p.dtype == torch.float32 for p in self.input.parameters())
        ):
            return False
        with torch.cuda.device(device):
            return not torch.cuda.is_current_stream_capturing()

    def forward_global_batch(
        self, global_batch, fake_quantize_acts=True, fake_quantize_weights=True, group=None,
        *, split_ft_backward=False,
    ):
        indices = self.layer_stacks.stack_remap[global_batch.layer_stack_indices]
        packed = global_batch.packed_payload
        x = self.input(
            global_batch.us,
            global_batch.them,
            global_batch.white_indices,
            global_batch.black_indices,
            fake_quantize_acts,
            fake_quantize_weights,
            **({
                "packed_payload": packed, "packed_feature_width": global_batch.packed_feature_width,
            } if packed is not None else {}),
        )
        if split_ft_backward:
            ft_output = x
            x = ft_output.detach().requires_grad_(True)
        l1 = self.layer_stacks.l1(
            x, indices, global_batch.offsets, fake_quantize_weights, group
        )
        prediction = self.layer_stacks.forward_from_l1(
            l1,
            indices[global_batch.local_slice],
            fake_quantize_acts,
            fake_quantize_weights,
        )
        return (prediction, (ft_output, x)) if split_ft_backward else prediction

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "ShardedNNUEModel requires forward_batch and a synchronized training step"
        )

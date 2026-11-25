"""Column-sharded feature transformer for sharded multi-GPU training."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from ...quantize import QuantizationManager
from ..features.input_feature import InputFeature
from .composed_feature_transformer import ComposedFeatureTransformer


@dataclass(frozen=True)
class FTShardLayout:
    """Paired FT columns and matching post-pairwise L1 input columns."""

    full_width: int
    world_size: int
    rank: int

    def __post_init__(self):
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError("rank must belong to a nonempty process group")
        if self.full_width <= 0 or self.full_width % (2 * self.world_size):
            raise ValueError(
                "FT width must be positive and divisible by twice the world size"
            )

    @property
    def pair_width(self) -> int:
        return self.full_width // (2 * self.world_size)

    @property
    def pair_slice(self) -> slice:
        start = self.rank * self.pair_width
        return slice(start, start + self.pair_width)

    @property
    def columns(self) -> tuple[slice, slice]:
        first = self.pair_slice
        half = self.full_width // 2
        return first, slice(half + first.start, half + first.stop)

    def select_columns(self, tensor: torch.Tensor) -> torch.Tensor:
        """Select this shard's two ranges from the last dimension."""
        if tensor.shape[-1] != self.full_width:
            raise ValueError("tensor's last dimension does not match the full FT width")
        first, second = self.columns
        return torch.cat((tensor[..., first], tensor[..., second]), dim=-1)


class ShardedFeatureTransformer(ComposedFeatureTransformer):
    """Own one paired-column FT shard; input positions must be global.

    The returned columns are [side-to-move pairs, opposite-side pairs]. Their
    locations in the unsharded FT output and L1 input are ``layout.columns``.
    This module performs no collectives and must not be wrapped in ordinary
    DDP, which would incorrectly reduce gradients of different FT shards.
    Construct on CPU before moving to the target device, as with the existing
    model; rank-specific initialization uses the CPU random generator.
    """

    _sharded_backward = True

    def __init__(
        self,
        feature_classes: list[Callable[[int], InputFeature]],
        full_width: int,
        quantization: QuantizationManager,
        *,
        rank: int,
        world_size: int,
    ):
        self.layout = FTShardLayout(full_width, world_size, rank)
        # DDP jobs commonly start every rank with the same seed. Consume one
        # value from the caller's CPU RNG, then initialize each rank from a
        # distinct seed without changing unrelated random streams. A freshly
        # initialized sharded model is reproducible, but does not reproduce the
        # exact random sequence of a newly initialized unsharded model.
        seed = int(
            torch.randint(0, 2**63 - 1, (), dtype=torch.int64, device="cpu").item()
        )
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed((seed + rank) % (2**63 - 1))
            super().__init__(feature_classes, 2 * self.layout.pair_width, quantization)
            self.init_weights()

    def get_extra_state(self):
        return {
            "version": 1,
            "full_width": self.layout.full_width,
            "world_size": self.layout.world_size,
            "rank": self.layout.rank,
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("FT checkpoint shard layout does not match this rank")

    @torch.no_grad()
    def load_unsharded(self, full: ComposedFeatureTransformer) -> None:
        """Copy paired columns from an existing, compatible full transformer.

        Constructing a shard directly does not allocate full-size FT weights;
        this optional conversion is intended for checkpoint migration and tests.
        """
        if (
            full.l1_size != self.layout.full_width
            or full.FEATURE_NAME != self.FEATURE_NAME
            or full.quantization.config != self.quantization.config
        ):
            raise ValueError("unsharded transformer configuration does not match")
        source = dict(full.named_parameters())
        target = dict(self.named_parameters())
        if source.keys() != target.keys():
            raise ValueError("unsharded transformer parameters do not match")
        # Check everything before copying so a mismatch cannot leave a partial load.
        for name, dest in target.items():
            src = source[name]
            if (
                src.shape[:-1] != dest.shape[:-1]
                or src.shape[-1] != self.layout.full_width
            ):
                raise ValueError(f"incompatible FT parameter shape for {name}")
        for name, dest in target.items():
            dest.copy_(self.layout.select_columns(source[name]))

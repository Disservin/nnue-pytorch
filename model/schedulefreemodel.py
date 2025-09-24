# from .lightning import NequIPLightningModule
from .lightning_module import NNUE
from typing import Dict, Any, Optional

# from nequip.utils import RankedLogger

# logger = RankedLogger(__name__, rank_zero_only=True)

# Note: Manual `.train()`/`.eval()` mode control for optimizer is required to ensure smoothed weights are captured at the right time
# Related discussion on Lightning timing hooks:
# https://github.com/Lightning-AI/pytorch-lightning/discussions/19759


class ScheduleFreeLightningModule(NNUE):
    """
    NequIP LightningModule using Facebook's Schedule-Free optimizer.

    This module wraps the model's optimizer in one of Facebook's Schedule-Free variants.
    See: https://github.com/facebookresearch/schedule_free

    Args:
        optimizer (Dict[str, Any]): Dictionary that must include a _target_
            corresponding to one of the Schedule-Free optimizers and other keyword arguments
            compatible with the Schedule-Free variants.
    """

    def __init__(self, **kwargs):
        self._schedulefree_state_dict: Dict[str, Any] = {}
        super().__init__(**kwargs)
        self._cached_opt = None  # will cache after trainer attaches

    def _sf_opt(self) -> Optional[Any]:
        if self._cached_opt is not None:
            return self._cached_opt
        if not hasattr(self, "trainer") or self.trainer is None:
            return None
        if not hasattr(self.trainer, "strategy"):
            return None
        opts = getattr(self.trainer.strategy, "optimizers", [])
        if not opts:
            return None
        self._cached_opt = opts[0]
        return self._cached_opt

    # --- checkpoint hooks ------------------------------------------------
    def on_save_checkpoint(self, checkpoint: dict):
        opt = self._sf_opt()
        if opt is not None:
            checkpoint["schedulefree_optimizer_state_dict"] = opt.state_dict()

    def on_load_checkpoint(self, checkpoint: dict):
        state = checkpoint.get("schedulefree_optimizer_state_dict")
        if state is not None:
            self._schedulefree_state_dict = state

     # --- fit lifecycle ---------------------------------------------------
    def on_fit_start(self) -> None:
        opt = self._sf_opt()
        if opt and hasattr(opt, "train"):
            opt.train()
            # (Re)load saved SF state if present (lazy)
            if self._schedulefree_state_dict:
                try:
                    opt.load_state_dict(self._schedulefree_state_dict)
                except Exception:
                    pass

    def on_train_start(self) -> None:
        opt = self._sf_opt()
        if opt and hasattr(opt, "train"):
            opt.train()

    def on_train_epoch_start(self) -> None:
        opt = self._sf_opt()
        if opt and hasattr(opt, "train"):
            opt.train()

    # --- validation ------------------------------------------------------
    def on_validation_epoch_start(self) -> None:
        opt = self._sf_opt()
        if opt and hasattr(opt, "eval"):
            opt.eval()

    def on_validation_epoch_end(self) -> None:
        # restore train mode for subsequent training epochs
        opt = self._sf_opt()
        if opt and hasattr(opt, "train"):
            opt.train()

    # --- test ------------------------------------------------------------
    def on_test_epoch_start(self) -> None:
        opt = self._sf_opt()
        if opt and hasattr(opt, "eval"):
            opt.eval()

    def on_test_epoch_end(self) -> None:
        opt = self._sf_opt()
        if opt and hasattr(opt, "train"):
            opt.train()

    # --- predict ---------------------------------------------------------
    def on_predict_start(self) -> None:
        opt = self._sf_opt()
        if opt and hasattr(opt, "eval"):
            opt.eval()

    def on_predict_end(self) -> None:
        opt = self._sf_opt()
        if opt and hasattr(opt, "train"):
            opt.train()

    # --- optimizer step safety -------------------------------------------
    def on_before_optimizer_step(self, optimizer):
        if hasattr(optimizer, "train"):
            optimizer.train()

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure=None,
    ):
        if hasattr(optimizer, "train"):
            optimizer.train()
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

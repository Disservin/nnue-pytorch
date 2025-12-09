import torch

from .serialize import NNUEReader
from ..config import ModelConfig
from ..features import FeatureSet
from ..model import NNUEModel
from ..quantize import QuantizationConfig


def load_model(
    filename: str,
    feature_set: FeatureSet,
    config: ModelConfig,
    quantize_config: QuantizationConfig,
    num_psqt_buckets: int | None = None,
) -> NNUEModel:
    if num_psqt_buckets is None:
        num_psqt_buckets = feature_set.get_default_num_psqt_buckets()

    if filename.endswith(".pt"):
        model = torch.load(filename, weights_only=False)
        model.eval()
        return model.model

    elif filename.endswith(".ckpt"):
        from ..lightning_module import NNUE

        model = NNUE.load_from_checkpoint(
            filename,
            feature_set=feature_set,
            config=config,
            quantize_config=quantize_config,
            num_psqt_buckets=num_psqt_buckets,
        )
        model.eval()
        return model.model

    elif filename.endswith(".nnue"):
        with open(filename, "rb") as f:
            reader = NNUEReader(
                f,
                feature_set,
                config,
                quantize_config,
                num_psqt_buckets=num_psqt_buckets,
            )
        return reader.model

    else:
        raise Exception("Invalid filetype: " + str(filename))

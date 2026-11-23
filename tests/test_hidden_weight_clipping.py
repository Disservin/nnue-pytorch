"""The hidden layers' weight clipping: the compiled version used on CUDA clamps exactly as the eager
one does, including the virtual (factorized) l1 weights and the layer-stack group factors."""

import pytest
import torch

from model.config import ModelConfig
from model.model import (
    NNUEModel,
    _clip_hidden_weights,
    _clip_hidden_weights_compiled,
)


def _hidden_weights(model):
    return [p for group in model.weight_clipping for p in group["params"]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("ls_factors", ["piececount+queen:pawnhalf:manyoffi", ""])
def test_compiled_clipping_matches_eager_bitwise(ls_factors):
    torch.manual_seed(0)
    model = NNUEModel(
        "Full_Threats+PP_3Wide+K32H3^", ModelConfig(L1=64, ls_factors=ls_factors)
    ).cuda()
    with torch.no_grad():
        # Push everything well outside the quantization range.
        for group in model.weight_clipping:
            for p in group["params"]:
                p.normal_(0, 3.0)
            if "virtual_params" in group:
                group["virtual_params"].normal_(0, 1.0)
            for fw in group["stacked_layer"].group_factor_w:
                fw.normal_(0, 1.0)
    before = [p.detach().clone() for p in _hidden_weights(model)]

    with torch.no_grad():
        _clip_hidden_weights(model.weight_clipping)
    eager = [p.detach().clone() for p in _hidden_weights(model)]

    with torch.no_grad():
        for p, b in zip(_hidden_weights(model), before):
            p.copy_(b)
        _clip_hidden_weights_compiled(model.weight_clipping)
    compiled = [p.detach().clone() for p in _hidden_weights(model)]

    assert any(not torch.equal(e, b) for e, b in zip(eager, before))
    for e, c in zip(eager, compiled):
        assert torch.equal(e, c)

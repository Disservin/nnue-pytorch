from types import SimpleNamespace

import pytest
import torch
from torch import nn

from model.config import LossParams
from model.lambda_utils import LambdaController
from model.nnue import NNUE, calculate_sf_loss, calculate_sf_loss_components
from model.sharded_backward import global_weighted_loss


def test_loss_components_preserve_nnue_loss_and_empty_graph():
    model = NNUE.__new__(NNUE)
    nn.Module.__init__(model)
    model.model = SimpleNamespace(quantization=SimpleNamespace(nnue2score=600.0))
    model.config = SimpleNamespace(loss_params=LossParams(w1=10.0))
    model.max_steps = 100
    model.lambda_scheduler = LambdaController()
    model._sf_loss_fn = calculate_sf_loss
    model._sf_loss_components_fn = calculate_sf_loss_components
    prediction = torch.tensor([[0.4], [-0.1]], requires_grad=True)
    scores = torch.tensor([[500.0], [-1000.0]])
    outcomes = torch.tensor([[1.0], [0.0]])
    batch = (None, None, None, None, outcomes, scores, None)
    expected = model.compute_loss_with_scorenet(prediction, batch, 4)
    numerator, denominator = model.compute_loss_components_with_scorenet(
        prediction, batch, 4
    )
    torch.testing.assert_close(numerator / denominator, expected)
    empty = prediction[:0]
    empty_batch = (None, None, None, None, outcomes[:0], scores[:0], None)
    numerator, denominator = model.compute_loss_components_with_scorenet(
        empty, empty_batch, 4
    )
    assert denominator == 0
    numerator.backward()
    torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))
    with pytest.raises(ValueError, match="finite and positive"):
        global_weighted_loss(numerator, denominator)

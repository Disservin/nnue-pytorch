"""The comparison benchmark's dense-to-packed conversion preserves inputs."""
import torch

from data_loader.distributed_features import _unpack_rows_reference
from scripts.benchmark_sharded_adamw import pack_saved_batch
from scripts.benchmark_sharded_adamw import match_sharded_initial_parameters
from model.nnue import NNUE
import model.model as model_module
import model.sharded_model as sharded_model_module
from test_sharded_adamw import _config
from test_sharded_ft import _Feature


def test_matched_benchmark_batch_conversion():
    white = torch.tensor([[1, 7, -1, -1], [4, 9, 12, -1], [-1, -1, -1, -1]], dtype=torch.int32)
    black = torch.tensor([[5, 8, 10, -1], [-1, -1, -1, -1], [0, 3, -1, -1]], dtype=torch.int32)
    batch = (torch.tensor([[1.], [0.], [1.]]), torch.tensor([[0.], [1.], [0.]]),
             white, black, torch.tensor([[0.], [0.5], [1.]]),
             torch.tensor([[-10.], [0.], [50.]]), torch.tensor([3, 1, 0]))
    result = pack_saved_batch(batch)
    native = result[2]
    n, width, nw, nb = (native[k] for k in ("size", "width", "nw", "nb"))
    data = native["packed"]
    wo, bo = data[:n + 1], data[n + 1:2 * (n + 1)]
    pos = 2 * (n + 1)
    white_decoded = _unpack_rows_reference(wo[1:] - wo[:-1], data[pos:pos + nw], width)
    pos += nw
    black_decoded = _unpack_rows_reference(bo[1:] - bo[:-1], data[pos:pos + nb], width)
    pos += nb
    torch.testing.assert_close(white_decoded, white, rtol=0, atol=0)
    torch.testing.assert_close(black_decoded, black, rtol=0, atol=0)
    torch.testing.assert_close(data[pos:].long(), batch[6], rtol=0, atol=0)
    for idx in (0, 1, 4, 5, 6):
        torch.testing.assert_close(result[idx], batch[idx], rtol=0, atol=0)


def test_matched_sharded_initial_parameters(monkeypatch):
    monkeypatch.setattr(model_module, "get_feature_cls", lambda _: [_Feature])
    monkeypatch.setattr(sharded_model_module, "get_feature_cls", lambda _: [_Feature])
    config, seed = _config(), 938
    torch.manual_seed(seed)
    full = NNUE(config, max_epoch=3, num_batches_per_epoch=5)
    shard = NNUE(config, max_epoch=3, num_batches_per_epoch=5, sharding=(1, 4))
    match_sharded_initial_parameters(shard, config, max_epoch=3, num_batches_per_epoch=5, seed=seed)
    reference = dict(full.model.named_parameters())
    for name, p in shard.model.named_parameters():
        expected = reference[name]
        if name.startswith("input.") or (name.startswith("layer_stacks.l1.") and p.ndim == 2):
            expected = shard.model.input.layout.select_columns(expected)
        torch.testing.assert_close(p, expected, rtol=0, atol=0)

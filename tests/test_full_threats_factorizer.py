"""Check the Full_Threats^ factorizer: its index table, export and clipping."""

import io
import os
import random
import sys

import chess
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig
from model.model import NNUEModel
from model.modules.features import DEFAULT_FEATURES
from model.modules.features.full_threats import (
    FullThreats,
    FullThreatsFactorized,
    threat_feature_table,
)
from model.utils.serialize import NNUEReader, NNUEWriter

FACTORIZED_FEATURES = DEFAULT_FEATURES.replace("Full_Threats", "Full_Threats^", 1)


def test_table_is_injective_and_maps_to_virtual_rows():
    entries = threat_feature_table()  # raises if two threats share an index
    assert len(entries) == 53312

    f = FullThreatsFactorized(8)
    vi = f.virtual_index
    assert vi.min() >= 0 and vi.max() < f.NUM_INPUTS_VIRTUAL
    # Each real feature maps to the row of its own (attacker, attacked, to).
    for attacker, _frm, to, attacked, index in entries[::97]:
        assert vi[index] == (attacker * 10 + attacked) * 64 + to
    unused = sorted(set(range(f.NUM_INPUTS)) - {e[4] for e in entries})
    assert len(unused) == f.NUM_INPUTS - len(entries)
    assert torch.all(vi[unused] == f.UNUSED_ROW)


def _python_threat_indices(board: chess.Board, perspective: chess.Color) -> set[int]:
    """Threat indices from python-chess, using threat_feature_table()."""
    lookup = {(a, fr, to, b): idx for a, fr, to, b, idx in threat_feature_table()}
    ksq = board.king(perspective)
    orient = (7 if chess.square_file(ksq) < 4 else 0) ^ (
        56 if perspective == chess.BLACK else 0
    )
    out = set()
    for frm, pc in board.piece_map().items():
        if pc.piece_type == chess.KING:
            continue
        for to in board.attacks(frm):
            target = board.piece_at(to)
            if target is None or target.piece_type == chess.KING:
                continue
            if pc.piece_type == chess.PAWN and target.piece_type == chess.PAWN:
                continue

            def tid(p: chess.Piece) -> int:
                color = 0 if p.color == chess.WHITE else 1
                if perspective == chess.BLACK:
                    color ^= 1
                return color * 5 + (p.piece_type - 1)

            key = (tid(pc), frm ^ orient, to ^ orient, tid(target))
            if key in lookup:
                out.add(lookup[key])
    return out


def _random_fens(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    fens = []
    while len(fens) < n:
        board = chess.Board()
        for _ in range(rng.randrange(4, 120)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if not board.is_game_over():
            fens.append(board.fen())
    return fens


def test_table_matches_data_loader():
    try:
        from data_loader import stream
    except (ImportError, OSError) as e:
        pytest.skip(f"data loader not available: {e}")

    fens = _random_fens(300, seed=1)
    batch = stream.get_sparse_batch_from_fens(
        "Full_Threats", fens, [0] * len(fens), [1] * len(fens), [0] * len(fens)
    )
    try:
        _, _, white, black, *_ = batch.contents.get_tensors("cpu")
        white, black = white.clone(), black.clone()
    finally:
        stream.destroy_sparse_batch(batch)

    for i, fen in enumerate(fens):
        board = chess.Board(fen)
        for color, rows in ((chess.WHITE, white), (chess.BLACK, black)):
            loader = {int(x) for x in rows[i] if x >= 0}
            assert loader == _python_threat_indices(board, color), (fen, color)


def _quantization():
    return NNUEModel(feature_name=DEFAULT_FEATURES, config=ModelConfig()).quantization


def test_export_coalesce_and_load():
    torch.manual_seed(0)
    f = FullThreatsFactorized(16)
    with torch.no_grad():
        f.weight.normal_()
        f.virtual_weight.normal_()
    expected = f.weight + f.virtual_weight[f.virtual_index]

    assert torch.equal(f.merged_weight(), expected)
    assert torch.equal(f.get_export_weights(), expected)

    f.coalesce()
    assert torch.count_nonzero(f.virtual_weight) == 0
    assert torch.allclose(f.weight, expected)

    g = FullThreatsFactorized(16)
    g.load_export_weights(expected)
    assert torch.equal(g.merged_weight(), expected)


def test_clip_keeps_virtual_part_and_bounds_merged():
    torch.manual_seed(1)
    q = _quantization()
    lo, hi = q.get_ft_weight_bounds(FullThreats.EXPORT_WEIGHT_DTYPE)
    f = FullThreatsFactorized(8)
    with torch.no_grad():
        f.weight.uniform_(3 * lo, 3 * hi)
        f.virtual_weight.uniform_(lo, hi)
    virtual_before = f.virtual_weight.clone()
    unclipped = f.merged_weight().detach().clone()

    f.clip_weights(q)

    merged = f.merged_weight().detach()
    assert torch.equal(f.virtual_weight, virtual_before)
    assert merged.min() >= lo - 1e-6 and merged.max() <= hi + 1e-6
    assert torch.allclose(merged, unclipped.clamp(lo, hi), atol=1e-6)


def test_gradient_reaches_virtual_rows():
    f = FullThreatsFactorized(4)
    f.merged_weight()[123].sum().backward()
    row = int(f.virtual_index[123])
    assert f.virtual_weight.grad[row].abs().sum() > 0
    assert torch.count_nonzero(f.virtual_weight.grad.abs().sum(dim=1)) == 1


def test_net_file_matches_unfactorized_features():
    torch.manual_seed(2)
    config = ModelConfig()
    model = NNUEModel(feature_name=FACTORIZED_FEATURES, config=config)
    threats = model.input.features[0]
    assert isinstance(threats, FullThreatsFactorized)
    assert model.feature_hash == NNUEModel(DEFAULT_FEATURES, config).feature_hash
    with torch.no_grad():
        threats.virtual_weight.uniform_(-0.05, 0.05)
    model.clip_weights(include_input=True)

    data = bytes(NNUEWriter(model, description="f", verbose=False).buf)

    plain = NNUEReader(io.BytesIO(data), DEFAULT_FEATURES, config).model
    assert torch.allclose(
        plain.input.features[0].weight, threats.merged_weight(), atol=1 / 64
    )
    again = bytes(NNUEWriter(plain, description="f", verbose=False).buf)
    assert data == again
